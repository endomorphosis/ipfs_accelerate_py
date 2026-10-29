"""Exact reexecuted user-manager identity; unknown procfs remains denied."""
from pathlib import Path
import os
import tempfile
import unittest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge as Bridge


class UserManagerDeserialize(unittest.TestCase):
    def process(self,root,pid,*,pam=False,command=None,executable='/usr/lib/systemd/systemd'):
        p=root/str(pid);p.mkdir();uid=os.geteuid();parent=2121 if pam else 1;group=2121
        (p/'status').write_text(f"Name:\t{'(sd-pam)' if pam else 'systemd'}\nPid:\t{pid}\nPPid:\t{parent}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        fields=['S',str(parent),str(group),str(group),*['0']*15,'424242']
        (p/'stat').write_text(f"{pid} {'((sd-pam))' if pam else '(systemd)'} {' '.join(fields)}\n")
        (p/'cgroup').write_text(f'0::/user.slice/user-{uid}.slice/user@{uid}.service/init.scope\n')
        (p/'cmdline').write_bytes(command if command is not None else (b'(sd-pam)\0' if pam else b'/usr/lib/systemd/systemd\0--user\0--deserialize=17\0'))
        (p/'exe').symlink_to(executable)
        return p
    def recognized(self,root,p):
        return Bridge._exact_user_manager_process(root,p,workspace_bytes=b'/owned/native/workspace')
    def test_single_numeric_deserialize_preserves_manager_and_pam_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manager=self.process(root,2121);pam=self.process(root,2122,pam=True)
            self.assertTrue(self.recognized(root,manager));self.assertTrue(self.recognized(root,pam))
    def test_unknown_arguments_executable_or_control_tuple_remain_denied(self):
        for command,exe,alter in ((b'/usr/lib/systemd/systemd\0--user\0--deserialize=\0','/usr/lib/systemd/systemd',None),
            (b'/usr/lib/systemd/systemd\0--user\0--deserialize=abc\0','/usr/lib/systemd/systemd',None),
            (b'/usr/lib/systemd/systemd\0--user\0--deserialize=17\0--extra\0','/usr/lib/systemd/systemd',None),
            (b'/tmp/systemd\0--user\0--deserialize=17\0','/usr/lib/systemd/systemd',None),
            (None,'/tmp/fake-systemd',None),(None,'/usr/lib/systemd/systemd','cgroup'),(None,'/usr/lib/systemd/systemd','parent')):
            with self.subTest(command=command,exe=exe,alter=alter),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);manager=self.process(root,2121,command=command,executable=exe)
                if alter=='cgroup':(manager/'cgroup').write_text('0::/different.scope\n')
                if alter=='parent':(manager/'status').write_text((manager/'status').read_text().replace('PPid:\t1','PPid:\t99'))
                self.assertFalse(self.recognized(root,manager))
