"""A canonical validator reaps adopted exits while a nested group still waits."""
from pathlib import Path
import json
import shlex
import sys

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    build_validation_environment, validation_environment_for_runner,
)


def test_nested_owned_group_reaps_adopted_zombie_before_command_exit(tmp_path):
    child = r"""
import json,os,signal,time
from pathlib import Path
signal.signal(signal.SIGTERM,signal.SIG_IGN)
def birth(pid):
 raw=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'state':raw[0],'parent_pid':int(raw[1]),'process_group':int(raw[2]),'session':int(raw[3]),'start_ticks':int(raw[19])}
original=birth(os.getpid())
Path('nested-child-ready.json').write_text(json.dumps(original))
deadline=time.monotonic()+3
while os.getppid()==original['parent_pid'] and time.monotonic()<deadline:time.sleep(.01)
current=birth(os.getpid());reaper=birth(os.getppid())
argv=Path(f'/proc/{reaper["pid"]}/cmdline').read_bytes().split(b'\0')
Path('nested-child-adopted.json').write_text(json.dumps({'child':current,'reaper':reaper,'native_subreaper':any(v.endswith(b'/native_cli_subreaper.py') for v in argv)}))
os._exit(0)
"""
    leader = "import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-I','-B','-c',"+repr(child)+"]);deadline=time.monotonic()+3\nwhile not Path('nested-child-ready.json').exists() and time.monotonic()<deadline:time.sleep(.01)\nassert Path('nested-child-ready.json').exists()\n"
    sdk_root = Path(native.__file__).resolve().parents[3]
    paths = [str(sdk_root), *(str(p) for p in sys.path if p and Path(p).is_dir())]
    inner = "import sys;sys.path[:0]="+repr(paths)+"\n"+r"""
import json,os,time
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime as runtime
def birth(pid):
 try:raw=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
 except FileNotFoundError:return None
 return {'pid':pid,'state':raw[0],'parent_pid':int(raw[1]),'process_group':int(raw[2]),'session':int(raw[3]),'start_ticks':int(raw[19])}
report={'main':birth(os.getpid())}
def started(process):report['leader']=birth(process.pid)
try:
 with Path('nested-output.log').open('w') as capture:
  completed=runtime.run_process_group_stream([sys.executable,'-I','-B','-c',LEADER_CODE],cwd=Path.cwd(),stdout=capture,env=dict(os.environ),inherit_environment=False,timeout_seconds=5,termination_grace_seconds=.75,on_started=started)
 report['returncode']=completed.returncode
 report['group_quiesced']=True
except BaseException as exc:
 report['error_type']=type(exc).__name__
 report['group_quiesced']=False
finally:
 report['original_child']=json.loads(Path('nested-child-ready.json').read_text())
 report['adoption']=json.loads(Path('nested-child-adopted.json').read_text())
 report['child_after_nested_wait']=birth(report['original_child']['pid'])
 report['main_still_alive']=birth(os.getpid()) is not None
 Path('nested-observation.json').write_text(json.dumps(report,sort_keys=True))
if not report['group_quiesced']:raise SystemExit(1)
raise SystemExit(7)
""".replace('LEADER_CODE',repr(leader))
    program = tmp_path/'nested-validator.py'
    program.write_text(inner)
    command = 'python -I -B '+shlex.quote(str(program))
    executor = native.PortalImplementationDaemon._validation_command_runner
    environment = validation_environment_for_runner(build_validation_environment(),executor)
    result = executor(spec=type('Command',(),{'command':command,'raw_command':command})(),
        workspace_path=tmp_path,timeout_seconds=10,environment=environment)
    (tmp_path/'canonical-output.log').write_text(result.get('output',''))
    assert (tmp_path/'nested-observation.json').is_file(), result.get('output','')
    report = json.loads((tmp_path/'nested-observation.json').read_text())
    assert result['returncode'] == 7,report
    assert result['validation_python_launcher']['sealed'] is True
    assert report['group_quiesced'] is True and report['returncode'] == 0
    assert report['main_still_alive'] is True
    assert report['original_child']['process_group'] == report['leader']['pid']
    assert report['original_child']['parent_pid'] == report['leader']['pid']
    assert report['adoption']['child']['start_ticks'] == report['original_child']['start_ticks']
    assert report['adoption']['child']['parent_pid'] == report['adoption']['reaper']['pid']
    assert report['adoption']['native_subreaper'] is True
    assert report['child_after_nested_wait'] is None
    assert not Path('/proc/'+str(report['original_child']['pid'])).exists()
