import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from layout_observer import Observer,LayoutTransition

class Fake:
    def __init__(self):self.kind='absent';self.calls=[]
    async def exec(self,*args,**kwargs):
        command=kwargs.get('command','')
        if command.startswith('python3 -I -c '):
            node=dict(exists=False) if self.kind=='absent' else dict(exists=True,kind=self.kind)
            if self.kind=='symlink':node['link_target']='/opt/ipfs-supervisor/source'
            return SimpleNamespace(return_code=0,stdout=json.dumps(dict(schema='fixed-runtime-toolchains-layout@1',root=dict(exists=True,kind='directory'),toolchains=node)))
        self.calls.append(command)
        if command=='extract':self.kind='directory'
        if command.startswith('private setup'):self.kind='symlink'
        return SimpleNamespace(return_code=0,stdout='not logged private output')
    async def upload_file(self,*args,**kwargs):self.calls.append('upload')

class LayoutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name);self.env=Fake();self.observer=Observer(self.env,self.path,'fixed-container')
        async def fake_probe():
            value=await self.env.exec(command='python3 -I -c probe')
            return json.loads(value.stdout)
        original=self.env.exec
        async def fake_probe():return json.loads((await original(command='python3 -I -c probe')).stdout)
        self.observer.probe=fake_probe

    async def test_transition_stops_and_preserves_only_hashes(self):
        self.observer.install()
        try:
            await self.env.exec(command='extract')
            with self.assertRaises(LayoutTransition):await self.env.exec(command='private setup secret-content')
        finally:self.observer.restore()
        raw=(self.path/'layout-events.jsonl').read_text();self.assertNotIn('secret-content',raw)
        self.assertNotIn('not logged private output',raw)
        value=json.loads((self.path/'layout-transition.json').read_text())
        self.assertEqual(value['prior_kind'],'directory')
        self.assertEqual(value['confirmation']['toolchains']['kind'],'symlink')
        self.assertEqual(self.env.exec,self.observer.original_exec)

    async def test_upload_does_not_open_credentials(self):
        self.observer.install()
        try:await self.env.upload_file('/missing/private-auth','/private-target')
        finally:self.observer.restore()
        self.assertNotIn('private-auth',(self.path/'layout-events.jsonl').read_text())
        self.assertNotIn('private-target',(self.path/'layout-events.jsonl').read_text())

    async def test_existing_bad_layout_refuses_before_action(self):
        self.env.kind='regular';self.observer.install()
        try:
            with self.assertRaises(LayoutTransition):await self.env.exec(command='extract')
        finally:self.observer.restore()
        self.assertEqual(self.env.calls,[])

    async def test_event_bound(self):
        self.observer.events=256
        with self.assertRaises(ValueError):await self.observer.observe({},'before')

    async def test_real_probe_uses_raw_docker_without_shell(self):
        observer=Observer(self.env,self.path,'fixed-container')
        raw=json.dumps(dict(schema='fixed-runtime-toolchains-layout@1',root={},toolchains={}))
        with patch('layout_observer.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=raw)) as run:
            await observer.probe()
        self.assertEqual(run.call_args.args[0][:10],['docker','exec','--user','root','fixed-container','python3','-I','-S','-B','-c'])
        self.assertNotIn('shell',run.call_args.kwargs)

if __name__=='__main__':unittest.main()
