"""Real parent/child/slow-worker topology for the native maintenance API."""
from pathlib import Path
import json
import os
import select
import signal
import subprocess
import sys
import time

root=Path(sys.argv[1])
role=sys.argv[2]
mode=sys.argv[3]

def write(name,value):
    path=root/name
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)

if role=='worker':
    signal.signal(signal.SIGINT,lambda *_: write('unexpected-signal.json',{'signal':2}))
    signal.signal(signal.SIGTERM,lambda *_: write('unexpected-signal.json',{'signal':15}))
    while not (root/'finish-worker').exists():
        write('heartbeat.json',{'pid':os.getpid(),'parent':os.getppid(),'at':time.monotonic()})
        time.sleep(.05)
    write('worker-closed.json',{'pid':os.getpid(),'parent':os.getppid()})
    raise SystemExit(0)
if role=='owner':
    worker=subprocess.Popen([sys.executable,'-B',__file__,str(root),'worker',mode],start_new_session=True)
    write('owner.json',{'pid':os.getpid(),'parent':os.getppid(),'worker':worker.pid})
    worker.wait()
    write('owner-closed.json',{'pid':os.getpid(),'worker_returncode':worker.returncode})
    raise SystemExit(0)

from scripts import run_agent_supervisor_efficiency_state_hardening as native
from ipfs_accelerate_py.agent_supervisor.runtime.source_maintenance_drain import SourceMaintenanceDrain

(root/'unknown-callback').write_text('started_outcome_unknown')
(root/'retained-capsule').write_text('retained original source')
worker_descriptor=None
observations=[]

def admission(binding):
    assert (root/'unknown-callback').read_text()=='started_outcome_unknown'
    assert (root/'retained-capsule').is_file()
    if mode in {'permission','interrupt'} and not (root/'allow-observation').exists():
        if mode=='interrupt': raise KeyboardInterrupt('injected observer interruption')
        raise PermissionError('nondumpable observation remains unavailable')

def closed(binding):
    assert worker_descriptor is not None
    poller=select.poll()
    poller.register(worker_descriptor,select.POLLIN)
    assert poller.poll(0),'live unknown worker cannot be inferred closed from owner exit'
    assert (root/'owner-closed.json').is_file()
    assert (root/'worker-closed.json').is_file()
    admission(binding)
    if mode=='closed_refusal' and not (root/'allow-observation').exists():
        raise RuntimeError('separate native cohort closure is unverified')

def publish(row):
    if mode=='publish' and not (root/'allow-observation').exists():
        raise OSError('native observation sink unavailable')
    observations.append(row)
    write('observation.json',row)

drain=SourceMaintenanceDrain(admission_gate=admission,closed_gate=closed,publish=publish)

def forbidden_force(*args,**kwargs):
    write('forced-cleanup.json',{'args':repr(args)})
    raise AssertionError('maintenance reached ordinary forced cleanup')

native._terminate_dedicated_process_group=forbidden_force
native._signal_dedicated_process_group=forbidden_force

def delegated(config_path,*,implement,duration,source_maintenance,parent_stop_state):
    global worker_descriptor
    child=subprocess.Popen([sys.executable,'-B',__file__,str(root),'owner',mode],start_new_session=True)
    birth=native._dedicated_process_group_birth(child)
    drain.bind(child,start_time_ticks=birth,source_head='a'*40,source_tree='b'*40)
    while not (root/'owner.json').exists(): time.sleep(.01)
    owner=json.loads((root/'owner.json').read_text())
    worker_descriptor=os.pidfd_open(owner['worker'])
    drain.request()
    write('admitted.json',{'parent':os.getpid(),'owner':child.pid,'worker':owner['worker'],'source_head':'a'*40})
    try:
        if mode=='finalizer': raise OSError('injected body failure while drain incomplete')
        native._wait_for_sealed_owner_exit(child,birth,source_maintenance=drain,parent_stop_state=parent_stop_state)
        return int(child.returncode)
    finally:
        native._finish_source_maintenance_before_cleanup(drain)
        native._retire_sealed_owner_child(child,birth,source_maintenance=drain)
        drain.close()
        os.close(worker_descriptor)
        (root/'retained-capsule').unlink()
        write('cleanup.json',{'complete':drain.complete,'owner_returncode':child.poll()})

native._run_supervisor_with_retained_child=delegated
result=native.run_supervisor(root/'fixture-config',implement=False,duration=0,source_maintenance=drain)
write('returned.json',{'result':result})
