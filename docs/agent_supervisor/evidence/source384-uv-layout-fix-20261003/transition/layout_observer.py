"""Bounded fixed-runtime-path observation; command hashes only."""
import asyncio
import hashlib
import json
import subprocess
import time

PROBE = r'''
import json,os,stat
def node(path):
 try:
  s=os.lstat(path)
 except FileNotFoundError:return dict(exists=False)
 result=dict(exists=True,kind='symlink' if stat.S_ISLNK(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'regular' if stat.S_ISREG(s.st_mode) else 'other',mode=s.st_mode,uid=s.st_uid,gid=s.st_gid,device=s.st_dev,inode=s.st_ino,bytes=s.st_size)
 if path=='/opt/ipfs-supervisor/toolchains' and stat.S_ISLNK(s.st_mode):
  target=os.readlink(path)
  if len(target.encode())>4096:raise ValueError('link target exceeds observation bound')
  result['link_target']=target
 return result
print(json.dumps(dict(schema='fixed-runtime-toolchains-layout@1',root=node('/opt/ipfs-supervisor'),toolchains=node('/opt/ipfs-supervisor/toolchains')),sort_keys=True))
'''

class LayoutTransition(RuntimeError):pass

class Observer:
    def __init__(self, environment, output, container):
        self.environment=environment;self.output=output
        self.original_exec=environment.exec;self.original_upload=environment.upload_file
        self.calls=0;self.events=0;self.last_kind=None
        self.container=container

    async def probe(self):
        response=await asyncio.to_thread(subprocess.run,
            ['docker','exec','--user','root',self.container,'python3','-I','-S','-B','-c',PROBE],
            capture_output=True,text=True,timeout=10)
        raw=response.stdout or ''
        if response.returncode or len(raw.encode())>32768:raise ValueError('bounded layout probe failed')
        value=json.loads(raw)
        if value.get('schema')!='fixed-runtime-toolchains-layout@1':raise ValueError('wrong layout schema')
        return value

    async def observe(self, label, when):
        self.events+=1
        if self.events>256:raise ValueError('layout event bound exceeded')
        value=await self.probe()
        event=dict(sequence=self.events,call=label,when=when,at=time.time(),observed=value)
        with (self.output/'layout-events.jsonl').open('a') as stream:stream.write(json.dumps(event,sort_keys=True)+'\n')
        kind=value['toolchains'].get('kind','absent')
        if kind not in ('absent','directory'):
            # Repeat only the same fixed-path lstat/readlink while this container is alive.
            confirmation=await self.probe()
            record=dict(schema='runtime-layout-transition@1',prior_kind=self.last_kind,
                triggering_event=event,confirmation=confirmation,stopped_before_next_operation=True,
                source384_context_executed=False,provider_calls=0,credential_contents_recorded=False)
            (self.output/'layout-transition.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
            raise LayoutTransition('fixed runtime toolchains path became '+kind)
        self.last_kind=kind

    def label(self, kind, content):
        self.calls+=1
        return dict(number=self.calls,kind=kind,sha256=hashlib.sha256(content.encode()).hexdigest())

    async def execute(self, *args, **kwargs):
        command=kwargs.get('command',args[0] if args else '')
        if type(command) is not str:raise TypeError('string command expected')
        label=self.label('exec',command)
        await self.observe(label,'before')
        result=await self.original_exec(*args,**kwargs)
        await self.observe(label,'after')
        return result

    async def upload(self, *args, **kwargs):
        # Never open uploaded content here (which may be private authentication).
        target=str(kwargs.get('target_path',args[1] if len(args)>1 else ''))
        label=self.label('upload_target',target)
        await self.observe(label,'before')
        result=await self.original_upload(*args,**kwargs)
        await self.observe(label,'after')
        return result

    def install(self):
        self.environment.exec=self.execute;self.environment.upload_file=self.upload

    def restore(self):
        self.environment.exec=self.original_exec;self.environment.upload_file=self.original_upload
