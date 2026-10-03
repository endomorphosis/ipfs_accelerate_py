import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import codex_cache_candidate as c

class CodexCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.vendor=self.root/'home/.nvm/versions/node/v22.0.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/bin'
        self.target=self.root/'provider-bin';self.vendor.mkdir(parents=True);self.target.mkdir()
        self.bodies={name:(name+'\x00').encode()*1024 for name in c.PINS}
        for name,raw in self.bodies.items():
            for parent in (self.vendor,self.target):
                p=parent/name;p.write_bytes(raw);p.chmod(0o755)
        self.pins={name:hashlib.sha256(raw).hexdigest() for name,raw in self.bodies.items()}
        self.addCleanup(patch.stopall);patch.object(c,'PINS',self.pins).start()
        patch.object(c,'SOURCE_UID',os.getuid()).start();patch.object(c,'TARGET_UID',os.getuid()).start()
        self.receipt=dict(schema='native-codex-runtime-bundle@1',codex_version='0.158.0',provider_calls=0,
            files=[dict(name=n,source_sha256=h,sha256=h,uid=0,mode=0o755) for n,h in self.pins.items()],
            executable_checks={n:dict(returncode=0,stdout_sha256='0'*64) for n in self.pins})

    def test_actual_syscalls_exact_four_bytes_metadata_parity(self):
        paths=[parent/name for name in self.pins for parent in (self.vendor,self.target)]
        before={p:c.identity(p.stat()) for p in paths}
        result=c.advise(self.root,self.receipt)
        self.assertEqual({p:c.identity(p.stat()) for p in paths},before)
        for p in paths:self.assertEqual(p.read_bytes(),self.bodies[p.name])
        self.assertEqual(result['selected_files'],4);self.assertEqual(result['advised_files'],4)
        self.assertEqual(result['body_read_bytes'],2*sum(map(len,self.bodies.values())))
        self.assertEqual(result['body_reads_after_advice'],0);self.assertFalse(result['freed_bytes_claimed'])

    def test_wrong_and_extra_duplicate_receipt_rows_refuse_before_hint(self):
        variants=[]
        value=copy.deepcopy(self.receipt);value['files'][0]['sha256']='f'*64;variants.append(value)
        value=copy.deepcopy(self.receipt);value['files'].append(value['files'][0]);variants.append(value)
        value=copy.deepcopy(self.receipt);value['files'][1]=value['files'][0];variants.append(value)
        value=copy.deepcopy(self.receipt);value['codex_version']='other';variants.append(value)
        value=copy.deepcopy(self.receipt);value['files'][0]['uid']=1000;variants.append(value)
        with patch.object(c.os,'posix_fadvise') as hint:
            for value in variants:
                with self.subTest(value=value),self.assertRaises(ValueError):c.advise(self.root,value)
            hint.assert_not_called()

    def test_body_corruption_refuses_before_any_hint(self):
        p=self.target/'codex-code-mode-host';raw=p.read_bytes();p.write_bytes(b'X'+raw[1:])
        with patch.object(c.os,'posix_fadvise') as hint,self.assertRaises(ValueError):c.advise(self.root,self.receipt)
        hint.assert_not_called()

    def test_parent_symlink_refused(self):
        parent=self.root/'home';parent.rename(self.root/'moved-home');parent.symlink_to(self.root/'moved-home')
        with patch.object(c.os,'posix_fadvise') as hint,self.assertRaises(OSError):c.advise(self.root,self.receipt)
        hint.assert_not_called()

    def test_leaf_symlink_hardlink_and_mode_refused(self):
        p=self.target/'codex';p.unlink();p.symlink_to(self.vendor/'codex')
        with self.assertRaises(OSError):c.advise(self.root,self.receipt)
        p.unlink();os.link(self.vendor/'codex',p)
        with self.assertRaises(ValueError):c.advise(self.root,self.receipt)
        p.unlink();p.write_bytes(self.bodies['codex']);p.chmod(0o644)
        with self.assertRaises(ValueError):c.advise(self.root,self.receipt)

    def test_size_and_owner_bounds(self):
        with patch.object(c,'MAX_FILE',1),self.assertRaises(ValueError):c.advise(self.root,self.receipt)
        with patch.object(c,'MAX_TOTAL',1),self.assertRaises(ValueError):c.advise(self.root,self.receipt)
        with patch.object(c,'TARGET_UID',os.getuid()+1),self.assertRaises(ValueError):c.advise(self.root,self.receipt)

    def test_ambiguous_vendor_refused(self):
        second=self.root/'home/.nvm/versions/node/v23.0.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-arm64/vendor/aarch64-unknown-linux-musl/bin'
        second.mkdir(parents=True);(second/'codex').write_bytes(b'other')
        with self.assertRaises(ValueError):c.advise(self.root,self.receipt)

    def test_no_reads_after_first_hint(self):
        original=c.os.read;hinted=False
        def read(fd,size):
            self.assertFalse(hinted);return original(fd,size)
        def hint(*args):
            nonlocal hinted
            hinted=True
        with patch.object(c.os,'read',side_effect=read),patch.object(c.os,'posix_fadvise',side_effect=hint):
            result=c.advise(self.root,self.receipt)
        self.assertEqual(result['advised_files'],4)

    def test_partial_read_and_metadata_drift_refused_before_hint(self):
        with patch.object(c.os,'read',return_value=b''),patch.object(c.os,'posix_fadvise') as hint:
            with self.assertRaises(ValueError):c.advise(self.root,self.receipt)
            hint.assert_not_called()
        original=c.os.read;changed=False
        def read(fd,size):
            nonlocal changed
            if not changed:(self.vendor/'codex').chmod(0o700);changed=True
            return original(fd,size)
        with patch.object(c.os,'read',side_effect=read),patch.object(c.os,'posix_fadvise') as hint:
            with self.assertRaises(ValueError):c.advise(self.root,self.receipt)
            hint.assert_not_called()

    def test_noatime_permission_failure_has_no_fallback(self):
        original=c.os.open
        def opened(path,flags,*args,**kwargs):
            if flags&c.os.O_NOATIME:raise PermissionError('no capability')
            return original(path,flags,*args,**kwargs)
        with patch.object(c.os,'open',side_effect=opened),self.assertRaises(PermissionError):c.advise(self.root,self.receipt)

    def test_hint_error_is_explicit_not_success(self):
        with patch.object(c.os,'posix_fadvise',side_effect=OSError(5,'failure')):
            result=c.advise(self.root,self.receipt)
        self.assertEqual(result['advised_files'],0);self.assertEqual(len(result['errors']),4)

    def test_receipt_leaf_guard_and_hash(self):
        p=self.root/'codex-cache-exposure.json';raw=json.dumps(self.receipt).encode();p.write_bytes(raw);p.chmod(0o644)
        pin=hashlib.sha256(raw).hexdigest()
        with patch.object(c,'RECEIPT_UID',os.getuid()):
            self.assertEqual(c.read_receipt(self.root,pin),self.receipt)
            with self.assertRaises(ValueError):c.read_receipt(self.root,'0'*64)
            os.link(p,self.root/'other')
            with self.assertRaises(ValueError):c.read_receipt(self.root,pin)
            (self.root/'other').unlink();p.chmod(0o600)
            with self.assertRaises(ValueError):c.read_receipt(self.root,pin)
            p.unlink();os.mkfifo(p)
            with self.assertRaises(ValueError):c.read_receipt(self.root,pin)

    def test_descriptors_closed_on_hash_failure(self):
        # Linux-only diagnostic already requires O_NOATIME/procfs.
        before=set(os.listdir('/proc/self/fd'))
        with patch.object(c.os,'read',return_value=b''),self.assertRaises(ValueError):c.advise(self.root,self.receipt)
        self.assertEqual(set(os.listdir('/proc/self/fd')),before)

    def test_exact_isolated_loader_binds_base_and_candidate(self):
        from run_diagnostic import native_cache_loader
        base=Path(c.__file__).with_name('cache_candidate.py')
        candidate=self.root/'small_loader_fixture.py'
        candidate.write_text("from cache_candidate import root_fd,identity\ndef protect_receipt(root,size):\n assert size==17\ndef main():\n print('exact-base-loaded')\n")
        args=[str(base),hashlib.sha256(base.read_bytes()).hexdigest(),str(candidate),
              hashlib.sha256(candidate.read_bytes()).hexdigest(),str(self.root/'receipt'),'0'*64,17]
        run=lambda values:subprocess.run([sys.executable,'-I','-S','-B','-c',native_cache_loader(*values)],capture_output=True,text=True,timeout=5)
        success=run(args);self.assertEqual(success.returncode,0,success.stderr)
        self.assertEqual(success.stdout.strip(),'exact-base-loaded')
        for index in (1,3):
            altered=list(args);altered[index]='f'*64;failed=run(altered)
            self.assertNotEqual(failed.returncode,0);self.assertNotIn('exact-base-loaded',failed.stdout)

    def test_receipt_protection_precedes_strict_read_and_preserves_identity(self):
        p=self.root/'codex-cache-exposure.json';raw=json.dumps(self.receipt).encode();p.write_bytes(raw);p.chmod(0o664)
        original=p.stat();pin=hashlib.sha256(raw).hexdigest()
        with patch.object(c,'RECEIPT_UID',os.getuid()):
            with self.assertRaises(ValueError):c.read_receipt(self.root,pin)
            result=c.protect_receipt(self.root,len(raw))
            self.assertEqual(result['body_reads'],0)
            self.assertEqual(c.read_receipt(self.root,pin),self.receipt)
        final=p.stat();self.assertEqual(original.st_ino,final.st_ino)
        self.assertEqual(original.st_atime_ns,final.st_atime_ns)
        self.assertEqual(original.st_mtime_ns,final.st_mtime_ns)
        self.assertEqual(final.st_mode&0o777,0o644)

    def test_receipt_protection_refuses_hardlink_fifo_size_before_metadata_write(self):
        p=self.root/'codex-cache-exposure.json';p.write_bytes(b'public');p.chmod(0o664)
        with patch.object(c.os,'fchown') as owner,patch.object(c.os,'fchmod') as mode:
            with self.assertRaises(ValueError):c.protect_receipt(self.root,7)
            os.link(p,self.root/'second')
            with self.assertRaises(ValueError):c.protect_receipt(self.root,6)
            p.unlink();os.mkfifo(p)
            with self.assertRaises(ValueError):c.protect_receipt(self.root,6)
            p.unlink();p.symlink_to(self.root/'second')
            with self.assertRaises(OSError):c.protect_receipt(self.root,6)
            owner.assert_not_called();mode.assert_not_called()

    def test_receipt_protection_detects_path_replacement(self):
        p=self.root/'codex-cache-exposure.json';p.write_bytes(b'public');p.chmod(0o664)
        original=c.os.fchmod
        def replace(fd,mode):
            original(fd,mode);p.rename(self.root/'prior');p.write_bytes(b'public');p.chmod(0o644)
        with patch.object(c,'RECEIPT_UID',os.getuid()),patch.object(c.os,'fchmod',side_effect=replace):
            with self.assertRaises(ValueError):c.protect_receipt(self.root,6)

    def test_binary_refusal_context_is_selected_metadata_only(self):
        p=self.target/'codex';p.chmod(0o644)
        with self.assertRaises(ValueError) as raised:c.advise(self.root,self.receipt)
        detail=json.loads(str(raised.exception).split(': ',1)[1])
        self.assertEqual(detail['path'],'provider-bin/codex')
        self.assertEqual(detail['mode'],0o644);self.assertEqual(detail['bytes'],len(self.bodies['codex']))

if __name__=='__main__':unittest.main()
