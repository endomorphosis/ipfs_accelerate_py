"""FD DATA regressions; alias/special-inode metadata cases are modeled.

No OS symlink, hardlink or special file is created. Ordinary parent replacement
is real. These tests grant no task completion, current health or runtime credit.
"""
import errno
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


class TestScientificEventEvidenceDATA(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='legal-ir-event-DATA-')
        self.root=Path(self.temp.name);self.parent=self.root/'event-data';self.parent.mkdir(mode=0o700)
        self.path=self.parent/'record.json';self.body=b'{"manifest":"event-bound","value":17}\n'
        self.path.write_bytes(self.body);self.path.chmod(0o664);self.h=executor.health_owner()
        self.ref=dict(path=str(self.path),bytes=len(self.body),sha256=hashlib.sha256(self.body).hexdigest())

    def tearDown(self):self.temp.cleanup()

    def read(self,ref=None):return executor._catalogue_race_event_evidence_data(self.ref if ref is None else ref)

    def leaf(self,fd):
        try:return os.readlink('/proc/self/fd/'+str(fd))==str(self.path)
        except OSError:return False

    def changed_stat(self,original,**changes):
        fields=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
        values={key:getattr(original,key)for key in fields};values.update(changes);return SimpleNamespace(**values)

    def test_exact_group_writable_historical_DATA_bytes(self):
        for mode in (0o600,0o644,0o664):
            with self.subTest(mode=mode):
                self.path.chmod(mode);self.assertEqual(self.read(),self.body);self.assertEqual(stat.S_IMODE(self.path.stat().st_mode),mode)

    def test_zero_raw_manifest_is_bounded_without_JSON_assumption(self):
        self.path.write_bytes(b'');reference=dict(self.ref,bytes=0,sha256=hashlib.sha256(b'').hexdigest())
        self.assertEqual(self.read(reference),b'')

    def test_closed_descriptor_shape_and_exact_bounds_refuse(self):
        cases=[dict(self.ref,extra=True),{k:v for k,v in self.ref.items()if k!='sha256'},dict(self.ref,bytes=True),
            dict(self.ref,bytes=-1),dict(self.ref,bytes=self.h.MAX_JSON+1),dict(self.ref,sha256='F'*64),
            dict(self.ref,sha256='0'*63),dict(self.ref,sha256='g'*64)]
        for reference in cases:
            with self.subTest(reference=reference):
                with self.assertRaises((executor.ExecutorHold,self.h.ExecutorHealthError)):self.read(reference)

    def test_same_size_wrong_digest_refuses(self):
        with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_changed'):self.read(dict(self.ref,sha256='0'*64))

    def test_wrong_declared_size_refuses(self):
        for delta in (-1,1):
            with self.subTest(delta=delta):
                with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_file_unsafe'):self.read(dict(self.ref,bytes=len(self.body)+delta))

    def test_world_writable_DATA_refuses_without_normalizing_history(self):
        for mode in (0o602,0o666):
            with self.subTest(mode=mode):
                self.path.chmod(mode)
                try:
                    with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_file_unsafe'):self.read()
                    self.assertEqual(stat.S_IMODE(self.path.stat().st_mode),mode)
                finally:self.path.chmod(0o600)  # owned synthetic fixture only

    def test_modeled_hardlink_UID_and_nonregular_leaf_refuse(self):
        original=executor.os.fstat
        for changes in ({'st_nlink':2},{'st_uid':os.geteuid()+1},{'st_mode':stat.S_IFIFO|0o600}):
            with self.subTest(model=changes):
                def modeled(fd):
                    observed=original(fd)
                    return self.changed_stat(observed,**changes)if self.leaf(fd)else observed
                with patch.object(executor.os,'fstat',side_effect=modeled):
                    with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_file_unsafe'):self.read()

    def test_modeled_leaf_fd_growth_refuses(self):
        original=executor.os.fstat;calls=0
        def modeled(fd):
            nonlocal calls
            observed=original(fd)
            if self.leaf(fd):
                calls+=1
                if calls>1:return self.changed_stat(observed,st_size=observed.st_size+1)
            return observed
        with patch.object(executor.os,'fstat',side_effect=modeled):
            with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_changed'):self.read()

    def test_same_inode_body_mutation_during_read_refuses(self):
        original=executor.os.read;changed=False
        def mutate(fd,count):
            nonlocal changed
            raw=original(fd,count)
            if self.leaf(fd)and raw and not changed:
                changed=True;self.path.write_bytes(b'x'*len(self.body))
            return raw
        with patch.object(executor.os,'read',side_effect=mutate):
            with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_changed'):self.read()

    def test_modeled_leaf_symlink_ELOOP_refuses(self):
        original=executor.os.open
        def modeled(path,flags,*args,**kwargs):
            if path==self.path.name and kwargs.get('dir_fd')is not None:raise OSError(errno.ELOOP,'modeled leaf alias')
            return original(path,flags,*args,**kwargs)
        with patch.object(executor.os,'open',side_effect=modeled):
            with self.assertRaises(OSError):self.read()

    def test_modeled_parent_symlink_metadata_refuses(self):
        original=executor.os.stat
        def modeled(path,*args,**kwargs):
            result=original(path,*args,**kwargs)
            if path==self.parent.name and kwargs.get('dir_fd')is not None:
                return self.changed_stat(result,st_mode=stat.S_IFLNK|0o777)
            return result
        with patch.object(executor.os,'stat',side_effect=modeled):
            with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_parent_alias'):self.read()

    def test_real_parent_replacement_during_read_refuses(self):
        original=executor.os.read;changed=False
        def replace(fd,count):
            nonlocal changed
            raw=original(fd,count)
            if self.leaf(fd)and raw and not changed:
                changed=True;self.parent.rename(self.root/'retained-original');self.parent.mkdir(mode=0o700);self.path.write_bytes(self.body)
            return raw
        with patch.object(executor.os,'read',side_effect=replace):
            with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_parent_changed'):self.read()
        self.assertEqual((self.root/'retained-original'/'record.json').read_bytes(),self.body)

    def test_all_opened_descriptors_close_on_hash_refusal(self):
        original_open=executor.os.open;original_close=executor.os.close;opened=[];closed=[]
        def opening(*args,**kwargs):
            fd=original_open(*args,**kwargs);opened.append(fd);return fd
        def closing(fd):closed.append(fd);return original_close(fd)
        with patch.object(executor.os,'open',side_effect=opening),patch.object(executor.os,'close',side_effect=closing):
            with self.assertRaises(executor.ExecutorHold):self.read(dict(self.ref,sha256='0'*64))
        self.assertTrue(opened);self.assertCountEqual(opened,closed)

    def test_real_parent_permission_change_during_read_refuses(self):
        original=executor.os.read;changed=False
        def mutate(fd,count):
            nonlocal changed
            raw=original(fd,count)
            if self.leaf(fd)and raw and not changed:changed=True;self.parent.chmod(0o755)
            return raw
        try:
            with patch.object(executor.os,'read',side_effect=mutate):
                with self.assertRaisesRegex(executor.ExecutorHold,'event_DATA_parent_changed'):self.read()
        finally:self.parent.chmod(0o700)

    def test_unrelated_parent_entry_change_preserves_bound_file(self):
        original=executor.os.read;changed=False
        def mutate(fd,count):
            nonlocal changed
            raw=original(fd,count)
            if self.leaf(fd)and raw and not changed:changed=True;(self.parent/'unrelated').write_bytes(b'ordinary')
            return raw
        with patch.object(executor.os,'read',side_effect=mutate):self.assertEqual(self.read(),self.body)

    def test_oversize_descriptor_refuses_before_any_open(self):
        with patch.object(executor.os,'open',side_effect=AssertionError('oversize must not open a file')):
            with self.assertRaises(executor.ExecutorHold):self.read(dict(self.ref,bytes=self.h.MAX_JSON+1))

    def test_modeled_named_generation_complete_direct_caller_accepts_and_refuses_foreign_file(self):
        # Only the named-root policy is modeled; real H documents, hashes,
        # directory digest and complete original caller logic all execute.
        root=self.root/'retained-generation';root.mkdir(mode=0o700)
        (root/'child').mkdir(mode=0o700);(root/'child'/'empty').mkdir(mode=0o700)
        rows=[]
        for relative,body in (('a',b'data'),('child/.hidden',b'hidden'),('child/zero',b'')):
            target=root/relative;target.write_bytes(body);target.chmod(0o600)
            rows.append(dict(relative_path=relative,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),mode=0o600))
        def write_control(name,value):
            target=self.root/name;body=(json.dumps(value,sort_keys=True)+'\n').encode();target.write_bytes(body);target.chmod(0o400)
            return dict(path=str(target),bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        manifest=write_control('generation-manifest.json',dict(schema='agent-supervisor/scientific-proof-file-manifest@1',root=str(root),files=rows))
        directories=[('.',0o700),('child',0o700),('child/empty',0o700)]
        value=dict(schema='agent-supervisor/scientific-proof-generation-census@1',root=str(root),manifests=[manifest],
            file_count=len(rows),source_bytes=sum(row['bytes']for row in rows),
            directory_sha256=hashlib.sha256(json.dumps(sorted(directories),separators=(',',':')).encode()).hexdigest())
        reference=write_control('generation-census.json',value)
        with patch.object(executor,'ORIGINAL_NAMED_ROOTS',(self.root,)):
            self.assertEqual(executor._generation_census(reference),value)
            (root/'foreign').write_bytes(b'unlisted');(root/'foreign').chmod(0o600)
            with self.assertRaisesRegex(executor.ExecutorHold,'generation_census_changed'):
                executor._generation_census(reference)
