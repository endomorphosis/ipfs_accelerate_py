"""Reusable timed runner for multiple implementation supervisor scripts."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

_UTC = timezone.utc  # noqa: UP017 - package supports Python 3.8.

# A datasets-authoritative configured-board process must not import repository
# code before its complete dependency closure is available as one immutable
# capsule.  Reject direct script/FD births before package-path restoration or
# any repository import.  Programmatic entry points repeat this decision.
_CONFIGURED_BOARD_LIVE_LAUNCH_FLAG = "--require-configured-board-live-seal"
_CONFIGURED_BOARD_LIVE_BIRTH_MARKER = (
    "--run-configured-board-live-seal-launch-gate"
)
if __package__ in {None, ""} and (
    _CONFIGURED_BOARD_LIVE_LAUNCH_FLAG in sys.argv[1:]
    or _CONFIGURED_BOARD_LIVE_BIRTH_MARKER in sys.argv[1:]
):
    raise SystemExit(78)

if __package__ in {None, ""}:
    # ``python -I /accepted/tree/.../multi_supervisor_runner.py`` excludes
    # ambient cwd, user-site, and PYTHONPATH authority.  Restore only this
    # file's accepted repository root before resolving package imports.
    _ACCEPTED_PACKAGE_ROOT = Path(__file__).absolute().parents[3]
    sys.path.insert(0, str(_ACCEPTED_PACKAGE_ROOT))
    __package__ = "ipfs_accelerate_py.agent_supervisor.runtime"

from ...llm_router import (
    AgentImplementationControlPlanePin,
    AgentSupervisorNativeDependencyLaunch,
    active_agent_supervisor_native_dependency_launch,
    build_agent_implementation_control_plane_pin,
    load_agent_implementation_route_authorization,
    resolve_agent_implementation_route,
    parse_agent_supervisor_native_dependency_launch,
    verify_agent_supervisor_native_dependency_sealed_fd,
    verify_agent_implementation_sealed_control_plane,
)
from ..._hash_resources import hashing_lock
from ..control.lifecycle_orchestrator import (
    CONFIGURATION_ROOT_ENV,
    FENCING_EPOCH_ENV,
    PROFILE_ID_ENV,
    REPOSITORY_ROOT_ENV,
    RUN_ID_ENV,
    RUN_ROOT_ENV,
    STATE_ROOT_ENV,
    TARGET_ID_ENV,
    LifecycleProfile,
    LinuxProcessAdapter,
    ProcessIdentity,
    ProcessIdentityMismatch,
    ProcessTreeSnapshot,
)
from ..core.wrapper_utils import (
    AgentSupervisorNamespacePaths,
    apply_env_defaults,
    env_str,
)
from ..merge.checkout_lock import serialized_lock_update
from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity as WorktreeProcessBirthIdentity,
    owner_liveness,
)
from ..proof.formal_verification_contracts import content_identity
from ..todo_daemon.core import (
    pid_alive,
    read_pid_file,
    remove_runtime_marker,
    terminate_pid_tree,
)
from ..todo_daemon.supervisor_runtime import (
    SupervisedChildIdentity,
    load_supervised_child_identity,
    read_process_command_argv,
    supervised_child_identity_liveness,
    supervised_child_identity_path,
)

OutputFn = Callable[[str], None]
PLAN_BOUND_LAUNCH_GATE_MARKER = "--run-plan-bound-launch-gate"
PLAN_BOUND_LAUNCH_GATE_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner"
)
PLAN_BOUND_LAUNCH_GATE_SUCCESS = b"\x01"
PLAN_BOUND_CHILD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/plan-bound-supervisor-child@1"
)
PLAN_BOUND_ACCEPTED_ENTRY_PATH = (
    "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
)
PLAN_BOUND_GATE_ENTRY_PATH = (
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py"
)
CONFIGURED_BOARD_LIVE_SEAL_PROFILE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-live-seal-profile@2"
)
CONFIGURED_BOARD_LIVE_SEAL_CHILD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/configured-board-live-seal-child@2"
)
CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_GATE_MARKER = (
    "--run-configured-board-live-seal-launch-gate"
)
CONFIGURED_BOARD_LIVE_SEAL_CONFIG_PATH = (
    "config/logic_governed_semantic_work_fabric_scheduler.json"
)
CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO = (
    "configured-board live multi-supervisor launch is NO-GO until the "
    "validator, verifier, scheduler, runner, target, and imported dependency "
    "closure are loaded from one immutable accepted control-plane capsule"
)
DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION = (
    "datasets-authoritative-operational-v1"
)
CONFIGURED_BOARD_LIVE_SEAL_VERIFIERS = MappingProxyType(
    {
        "scripts/validate_logic_governed_semantic_work_fabric_board.py": (
            "scripts/"
            "materialize_logic_governed_semantic_work_fabric_control_plane.py"
        ),
    }
)
PLAN_BOUND_REPLAN_RETURN_CODE = 75
STALE_DETACHED_MASTER_PID_DECISION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "stale-detached-master-pid-quarantine-decision@1"
)
STALE_DETACHED_MASTER_PID_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "stale-detached-master-pid-quarantine@1"
)
_LEGACY_MASTER_PID_PAYLOAD = re.compile(rb"[1-9][0-9]*\n")
_LEGACY_MASTER_PID_MAX_BYTES = 32
SEALED_CONTROL_PLANE_MODULES = frozenset(
    {
        "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
    }
)
SEALED_CONTROL_PLANE_BOOTSTRAP = r'''import array,ctypes,fcntl,hashlib,json,os,socket,stat,struct,sys,time
from contextlib import contextmanager
@contextmanager
def _hash_budget():
    # Same lock as _hash_resources, before any capsule code is trusted/imported.
    path='/tmp/ipfs-accelerate-heavy-hash-'+str(os.geteuid())+'.lock'
    lock=os.open(path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    acquired=False
    try:
        metadata=os.fstat(lock)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid!=os.geteuid() or metadata.st_nlink!=1 or stat.S_IMODE(metadata.st_mode)&0o022: raise SystemExit(78)
        deadline=time.monotonic()+60.0
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); acquired=True; break
            except BlockingIOError:
                if time.monotonic()>=deadline: raise SystemExit(78)
                time.sleep(0.05)
        current=os.stat(path,follow_symlinks=False)
        if (current.st_dev,current.st_ino)!=(metadata.st_dev,metadata.st_ino) or os.fstat(lock).st_nlink!=1: raise SystemExit(78)
        yield
    finally:
        if acquired: fcntl.flock(lock,fcntl.LOCK_UN)
        os.close(lock)
def _state_authority_handoff():
    names=('IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_ADDRESS','IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_PID','IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_START','IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_BOOT_ID','IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_LOSS_POLICY')
    values={name:os.environ.get(name,'').strip() for name in names}
    if not any(values.values()): return
    if '--run-plan-bound-launch-gate' in sys.argv[1:]: return
    if not all(values.values()) or not sys.platform.startswith('linux'): raise SystemExit(78)
    try:
        executable_fd=os.open('/proc/self/exe',getattr(os,'O_PATH',os.O_RDONLY)|getattr(os,'O_CLOEXEC',0))
        executable_metadata=os.fstat(executable_fd); executable_path_metadata=os.stat('/proc/self/exe')
    except OSError: raise SystemExit(78)
    if not stat.S_ISREG(executable_metadata.st_mode) or executable_metadata.st_uid!=0 or executable_metadata.st_nlink!=1 or stat.S_IMODE(executable_metadata.st_mode)&0o022 or (executable_metadata.st_dev,executable_metadata.st_ino,executable_metadata.st_mode,executable_metadata.st_uid,executable_metadata.st_gid,executable_metadata.st_nlink,executable_metadata.st_size)!=(executable_path_metadata.st_dev,executable_path_metadata.st_ino,executable_path_metadata.st_mode,executable_path_metadata.st_uid,executable_path_metadata.st_gid,executable_path_metadata.st_nlink,executable_path_metadata.st_size):
        os.close(executable_fd); raise SystemExit(78)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(4,0,0,0,0)!=0 or libc.prctl(3,0,0,0,0)!=0:
        os.close(executable_fd); raise SystemExit(78)
    try:
        parent=int(values[names[1]]); parent_start=int(values[names[2]])
        raw=open('/proc/self/stat','r',encoding='ascii').read(); fields=raw[raw.rfind(')')+2:].split()
        own_parent=int(fields[1]); own_start=int(fields[19])
        parent_raw=open('/proc/'+str(parent)+'/stat','r',encoding='ascii').read(); parent_fields=parent_raw[parent_raw.rfind(')')+2:].split()
        observed_parent_start=int(parent_fields[19]); boot=open('/proc/sys/kernel/random/boot_id','r',encoding='ascii').read().strip()
    except (OSError,IndexError,UnicodeError,ValueError): raise SystemExit(78)
    policy=values[names[4]]
    if own_parent!=parent or observed_parent_start!=parent_start or boot!=values[names[3]] or policy not in {'terminate_with_parent','independent_detached'}: raise SystemExit(78)
    if policy=='terminate_with_parent':
        if libc.prctl(1,15,0,0,0)!=0: raise SystemExit(78)
        try:
            after_raw=open('/proc/self/stat','r',encoding='ascii').read(); after_fields=after_raw[after_raw.rfind(')')+2:].split()
            parent_after=int(after_fields[1]); parent_raw_after=open('/proc/'+str(parent)+'/stat','r',encoding='ascii').read(); parent_fields_after=parent_raw_after[parent_raw_after.rfind(')')+2:].split()
            parent_start_after=int(parent_fields_after[19])
        except (OSError,IndexError,UnicodeError,ValueError): raise SystemExit(78)
        if parent_after!=parent or parent_start_after!=parent_start: raise SystemExit(78)
    channel=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); received=[]
    try:
        channel.settimeout(10.0); channel.connect('\0'+values[names[0]])
        peer=struct.unpack('3i',channel.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i')))
        if peer[0]!=parent or peer[1]!=os.geteuid(): raise SystemExit(78)
        request=json.dumps({'pid':os.getpid(),'parent_pid':parent,'start_time_ticks':own_start,'boot_id':boot,'address':values[names[0]],'parent_loss_policy':policy},sort_keys=True,separators=(',',':')).encode()+b'\n'
        descriptors=array.array('i',[executable_fd])
        channel.sendmsg([request],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,descriptors.tobytes())])
        os.close(executable_fd); executable_fd=-1
        data,ancillary,flags,_=channel.recvmsg(1,socket.CMSG_SPACE(array.array('i').itemsize))
        if data!=b'F' or flags&getattr(socket,'MSG_CTRUNC',0): raise SystemExit(78)
        for level,kind,payload in ancillary:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                descriptors=array.array('i'); descriptors.frombytes(payload[:len(payload)-(len(payload)%descriptors.itemsize)]); received.extend(descriptors)
        if len(received)!=1: raise SystemExit(78)
        secret_fd=int(received[0]); metadata=os.fstat(secret_fd)
        required=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SEAL
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid!=os.geteuid() or not 32<=metadata.st_size<=256 or fcntl.fcntl(secret_fd,fcntl.F_GET_SEALS)&required!=required: raise SystemExit(78)
        os.set_inheritable(secret_fd,False); os.environ['IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD']=str(secret_fd)
        for name in names: os.environ.pop(name,None)
        channel.sendall(b'A')
    except SystemExit: raise
    except BaseException: raise SystemExit(78)
    finally:
        if executable_fd>=0: os.close(executable_fd)
        channel.close()
def _pairs(items):
    result={}
    for key,value in items:
        if key in result: raise SystemExit(78)
        result[key]=value
    return result
try:
    fd=int(sys.argv.pop(1)); pin=json.loads(sys.argv.pop(1),object_pairs_hook=_pairs)
    native_authorization=sys.argv.pop(1); native_fd=int(sys.argv.pop(1)); native_text=sys.argv.pop(1); system_text=sys.argv.pop(1)
    module=sys.argv.pop(1); expected_bootstrap=sys.argv.pop(1); expected_python=sys.argv.pop(1)
    if fd<3 or native_fd<3 or fd==native_fd or type(pin) is not dict or set(pin)!={'schema','runner_path','runner_sha256','capsule_root','capsule_id','source_head','source_tree','archive_sha256'}: raise SystemExit(78)
    if any(type(value) is not str or not value for value in pin.values()): raise SystemExit(78)
    if pin['schema']!='ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2': raise SystemExit(78)
    if module not in {'ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler','ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor'}: raise SystemExit(78)
    if not sys.flags.isolated or not sys.flags.no_site or any(name.startswith(('LD_','DYLD_','PYTHON','PYTEST')) or name=='GLIBC_TUNABLES' for name in os.environ): raise SystemExit(78)
    command_line=open('/proc/self/cmdline','rb').read().split(b'\0')
    code_index=command_line.index(b'-c')+1
    if 'sha256:'+hashlib.sha256(command_line[code_index]).hexdigest()!=expected_bootstrap: raise SystemExit(78)
    with _hash_budget():
        executable=os.open('/proc/self/exe',os.O_RDONLY|getattr(os,'O_CLOEXEC',0))
        try:
            executable_hash=hashlib.sha256()
            while True:
                block=os.read(executable,65536)
                if not block: break
                executable_hash.update(block)
        finally: os.close(executable)
        if 'sha256:'+executable_hash.hexdigest()!=expected_python: raise SystemExit(78)
        required=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SEAL
        metadata=os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size<=0 or fcntl.fcntl(fd,fcntl.F_GET_SEALS)&required!=required: raise SystemExit(78)
        archive_hash=hashlib.sha256(); offset=0
        while offset<metadata.st_size:
            block=os.pread(fd,min(65536,metadata.st_size-offset),offset)
            if not block: break
            archive_hash.update(block); offset+=len(block)
        if offset!=metadata.st_size or 'sha256:'+archive_hash.hexdigest()!=pin['archive_sha256']: raise SystemExit(78)
    archive='/proc/self/fd/'+str(fd)
    path_metadata=os.stat(archive)
    if (path_metadata.st_dev,path_metadata.st_ino)!=(metadata.st_dev,metadata.st_ino): raise SystemExit(78)
    try: system=json.loads(system_text,object_pairs_hook=_pairs)
    except BaseException: raise SystemExit(78)
    if type(system) is not list or json.dumps(system,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)!=system_text: raise SystemExit(78)
    expected_paths={'/usr/local/lib/python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'/dist-packages','/usr/lib/python3/dist-packages'}
    observed_paths=[]
    for item in system:
        if type(item) is not dict or set(item)!={'path','st_dev','st_ino','st_mode','st_uid','st_nlink','st_mtime_ns','st_ctime_ns'} or type(item['path']) is not str or item['path'] not in expected_paths or any(type(item[name]) is not int for name in set(item)-{'path'}): raise SystemExit(78)
        current=os.lstat(item['path']); identity=(current.st_dev,current.st_ino,current.st_mode,current.st_uid,current.st_nlink,current.st_mtime_ns,current.st_ctime_ns)
        if identity!=tuple(item[name] for name in ('st_dev','st_ino','st_mode','st_uid','st_nlink','st_mtime_ns','st_ctime_ns')) or not stat.S_ISDIR(current.st_mode) or current.st_uid!=0 or stat.S_IMODE(current.st_mode)&0o022 or os.path.realpath(item['path'])!=item['path']: raise SystemExit(78)
        observed_paths.append(item['path'])
    if len(observed_paths)!=len(set(observed_paths)) or '/usr/lib/python3/dist-packages' not in observed_paths: raise SystemExit(78)
    sys.path.insert(0,archive); sys.path.extend(observed_paths)
    import importlib,importlib.machinery,runpy,types
    import ipfs_accelerate_py as accepted_root
    prefix=archive+'/'
    root_origin=getattr(accepted_root,'__file__',None)
    if type(root_origin) is not str or not root_origin.startswith(prefix): raise SystemExit(78)
    from ipfs_accelerate_py.agent_implementation_route import _agent_parse_native_dependency_launch_json,preload_agent_supervisor_native_dependency,verify_agent_supervisor_native_dependency_sealed_fd
    native=_agent_parse_native_dependency_launch_json(native_text)
    if native.accepted_authorization_id!=native_authorization or native.descriptor.descriptor!=native_fd or native.pin.python_executable_sha256!=expected_python or verify_agent_supervisor_native_dependency_sealed_fd(native)!='/proc/self/fd/'+str(native_fd): raise SystemExit(78)
    preload_agent_supervisor_native_dependency(native)
    package_name='ipfs_accelerate_py.agent_supervisor'
    if any(name==package_name or name.startswith(package_name+'.') for name in sys.modules): raise SystemExit(78)
    package=importlib.import_module(package_name)
    package_origin=getattr(package,'__file__',None)
    if type(package_origin) is not str or not package_origin.startswith(prefix): raise SystemExit(78)
    setattr(accepted_root,'agent_supervisor',package)
    if module in sys.modules: raise SystemExit(78)
    specification=importlib.util.find_spec(module)
    module_origin=getattr(specification,'origin',None)
    if type(module_origin) is not str or not module_origin.startswith(prefix): raise SystemExit(78)
    _state_authority_handoff()
    namespace=runpy.run_module(module,run_name=module,alter_sys=True)
    if module in sys.modules: raise SystemExit(78)
    target_origin=namespace.get('__file__')
    if type(target_origin) is not str or not target_origin.startswith(prefix): raise SystemExit(78)
    for name,loaded in tuple(sys.modules.items()):
        if name in {'ipfs_accelerate_py','ipfs_accelerate_py.llm_router','ipfs_accelerate_py.agent_implementation_route','ipfs_accelerate_py._hash_resources'} or name.startswith('ipfs_accelerate_py.agent_supervisor'):
            origin=getattr(loaded,'__file__',None)
            if type(origin) is not str or not origin.startswith(prefix): raise SystemExit(78)
    main=namespace.get('main')
    if not callable(main): raise SystemExit(78)
    raise SystemExit(main())
except SystemExit: raise
except BaseException: raise SystemExit(78)
'''
SEALED_CONTROL_PLANE_BOOTSTRAP_SHA256 = (
    "sha256:"
    + hashlib.sha256(SEALED_CONTROL_PLANE_BOOTSTRAP.encode("utf-8")).hexdigest()
)


@dataclass(frozen=True)
class RetainedControlPlaneInterpreter:
    """Exact executable descriptor retained across authority-bearing exec."""

    descriptor: int
    argv0: str
    executable_path: str
    sha256: str
    identity: tuple[int, int, int, int, int, int, int, int]


SEALED_SYSTEM_DEPENDENCY_DIRS_ENV = (
    "IPFS_ACCELERATE_AGENT_SEALED_SYSTEM_DEPENDENCY_DIRS_JSON"
)
SEALED_NATIVE_DEPENDENCY_FD_ENV = (
    "IPFS_ACCELERATE_AGENT_SEALED_NATIVE_DEPENDENCY_FD"
)
SEALED_NATIVE_DEPENDENCY_LAUNCH_ENV = (
    "IPFS_ACCELERATE_AGENT_SEALED_NATIVE_DEPENDENCY_LAUNCH_JSON"
)

ORDERED_IMPLEMENTATION_PROVIDER_ROUTE: Mapping[str, str] = MappingProxyType(
    resolve_agent_implementation_route(default_route="legacy").as_environment()
)
_IMPLEMENTATION_PROVIDER_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
)
_ROUTE_AUTHORIZATION_ENV_NAMES = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID",
)
_PROVIDER_EXECUTABLE_ENV_NAMES = (
    "IPFS_ACCELERATE_AGENT_GROK_BIN",
)
PROVIDER_EXTERNAL_ISOLATION_ENV = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_EXTERNAL_ISOLATION_JSON"
)
_DOCKER_CLEANUP_WATCHDOG_ARG = "--internal-docker-cleanup-watchdog"
_DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG = (
    "--internal-docker-cleanup-watchdog-launcher"
)
_DOCKER_CLEANUP_CONTAINER_RE = re.compile(
    r"ipfs-accelerate-(?:grok|codex)-[0-9]+-[0-9a-f]{32}"
)
_DOCKER_CLEANUP_INSPECTION_MAX_BYTES = 256 * 1024
_DOCKER_LOCAL_HOST = "unix:///var/run/docker.sock"
_DOCKER_CLEANUP_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-cleanup-binding@6"
)
_DOCKER_CREATE_JOURNAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/docker-create-journal@4"
)
_DOCKER_CLEANUP_BINDING_DIRECTORY = "provider-cleanup-bindings"
_DOCKER_CREATE_JOURNAL_NAME = "create-journal.json"
_DOCKER_PRIVATE_CONTROL_MAX_BYTES = 512 * 1024
_DOCKER_CLEANUP_STABLE_ENTRY_RE = re.compile(
    r"[0-9a-f]{64}\.(?:json|authority|complete|lock|remove-dispatched)"
)
_DOCKER_CLEANUP_ATOMIC_TEMP_RE = re.compile(
    r"\.(?P<target>[0-9a-f]{64}\."
    r"(?:json|authority|complete|lock|remove-dispatched))\."
    r"(?P<nonce>[0-9a-f]{16})"
)
_DOCKER_CLEANUP_PUBLICATION_WAIT_SECONDS = 0.5
_DOCKER_CLEANUP_PUBLICATION_POLL_SECONDS = 0.01


@dataclass(frozen=True)
class _DurableDockerCleanupBinding:
    docker_bin: str
    provider: str
    container_name: str
    cleanup_root: Path
    cleanup_root_identity: Mapping[str, int]
    lease_root: Path
    docker_config: Path
    cidfile: Path
    provider_home: Path
    prompt_path: Path
    effect_observation: Mapping[str, str]
    create_command_id: str
    create_cwd: Path
    create_environment_id: str
    termination_fence: Mapping[str, object]
    binding_state: str
    path_identities: Mapping[str, Mapping[str, int]]
    runner_pid: int
    runner_start_ticks: int
    watchdog_pid: int
    watchdog_start_ticks: int
    boot_id: str
    record_path: Path
    record_device: int
    record_inode: int
    record_id: str

    @property
    def binding(self) -> tuple[str, str, str]:
        return self.docker_bin, self.container_name, str(self.lease_root)


@dataclass(frozen=True)
class _DurableCleanupDirectoryAnchor:
    """Parent-held identity for the cleanup binding namespace.

    The accepted runner creates and opens this directory before any child can
    publish a provider effect.  Shutdown scans use the retained descriptor and
    also require the public path to resolve to the same inode.  Renaming the
    directory therefore produces UNKNOWN instead of an apparently empty scan.
    """

    descriptor: int
    path: Path
    device: int
    inode: int
    mode: int
    uid: int


def _cleanup_directory_stat_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
    )


def _open_durable_cleanup_directory_anchor(
    run_root: Path,
) -> _DurableCleanupDirectoryAnchor:
    """Create and retain the one cleanup namespace used by a managed track."""

    resolved_run_root = Path(run_root)
    resolved_run_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if (
        resolved_run_root.resolve(strict=True) != resolved_run_root.absolute()
        or not stat.S_ISDIR(os.lstat(resolved_run_root).st_mode)
    ):
        raise ValueError("managed lifecycle run root is aliased")
    directory = resolved_run_root / _DOCKER_CLEANUP_BINDING_DIRECTORY
    directory.mkdir(mode=0o700, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(directory)
        if (
            directory.resolve(strict=True) != directory.absolute()
            or _cleanup_directory_stat_identity(opened)
            != _cleanup_directory_stat_identity(current)
            or not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise ValueError("durable Docker cleanup directory is not private")
        return _DurableCleanupDirectoryAnchor(
            descriptor=descriptor,
            path=directory,
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
            mode=int(opened.st_mode),
            uid=int(opened.st_uid),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _validate_durable_cleanup_directory_anchor(
    anchor: _DurableCleanupDirectoryAnchor,
    *,
    expected_path: Path,
) -> os.stat_result:
    """Require both the retained descriptor and public name to remain exact."""

    if anchor.path != expected_path:
        raise ValueError("durable Docker cleanup anchor path drifted")
    try:
        opened = os.fstat(anchor.descriptor)
        current = os.lstat(expected_path)
    except OSError as exc:
        raise ValueError("durable Docker cleanup anchor is unavailable") from exc
    expected = (anchor.device, anchor.inode, anchor.mode, anchor.uid)
    if (
        _cleanup_directory_stat_identity(opened) != expected
        or _cleanup_directory_stat_identity(current) != expected
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
        or expected_path.resolve(strict=True) != expected_path.absolute()
    ):
        raise ValueError("durable Docker cleanup anchor identity drifted")
    return opened


def _docker_control_identity(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class _SupportsFileno(Protocol):
    def fileno(self) -> int: ...


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_accepted_control_plane_pin(
    value: str | Mapping[str, object],
) -> AgentImplementationControlPlanePin:
    """Strictly decode and revalidate one public control-plane pin DTO."""

    if isinstance(value, str):
        try:
            payload = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("accepted control-plane pin is invalid JSON") from exc
    elif type(value) is dict:
        payload = dict(value)
    else:
        raise ValueError("accepted control-plane pin must be an exact object")
    expected = {
        "schema",
        "runner_path",
        "runner_sha256",
        "capsule_root",
        "capsule_id",
        "source_head",
        "source_tree",
        "archive_sha256",
    }
    if (
        type(payload) is not dict
        or set(payload) != expected
        or any(type(payload[name]) is not str or not payload[name] for name in expected)
    ):
        raise ValueError("accepted control-plane pin fields are not exact")
    pin = AgentImplementationControlPlanePin(**payload)
    verified = build_agent_implementation_control_plane_pin(
        runner_path=pin.runner_path,
        capsule_root=pin.capsule_root,
    )
    if verified != pin:
        raise ValueError("accepted control-plane pin changed during decode")
    return pin


def accepted_control_plane_pin_json(
    pin: AgentImplementationControlPlanePin,
) -> str:
    return json.dumps(
        pin.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _control_plane_interpreter_identity(
    item: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        int(item.st_dev),
        int(item.st_ino),
        int(item.st_mode),
        int(item.st_uid),
        int(item.st_nlink),
        int(item.st_size),
        int(item.st_mtime_ns),
        int(item.st_ctime_ns),
    )


def admit_retained_control_plane_interpreter(
    *,
    descriptor: int,
    argv0: str,
    expected_sha256: str,
) -> RetainedControlPlaneInterpreter:
    """Validate an inherited exact interpreter without reopening its name."""

    with hashing_lock(kind="trusted-executable", exclusive=True):
        return _admit_retained_control_plane_interpreter_unlocked(
            descriptor=descriptor, argv0=argv0, expected_sha256=expected_sha256
        )


def _admit_retained_control_plane_interpreter_unlocked(
    *, descriptor: int, argv0: str, expected_sha256: str,
) -> RetainedControlPlaneInterpreter:
    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 3
        or not isinstance(argv0, str)
        or not Path(argv0).is_absolute()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("retained control-plane interpreter binding is invalid")
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
        executable_path = f"/proc/self/fd/{descriptor}"
        proc = os.stat(executable_path)
    except OSError as exc:
        raise ValueError(
            "retained control-plane interpreter is unavailable"
        ) from exc
    identity = _control_plane_interpreter_identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != 0
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > 128 * 1024 * 1024
        or stat.S_IMODE(before.st_mode) & 0o022
        or stat.S_IMODE(before.st_mode) & 0o111 == 0
        or offset != before.st_size
        or _control_plane_interpreter_identity(after) != identity
        or (int(proc.st_dev), int(proc.st_ino)) != identity[:2]
        or "sha256:" + digest.hexdigest() != expected_sha256
    ):
        raise ValueError("retained control-plane interpreter identity drifted")
    return RetainedControlPlaneInterpreter(
        descriptor=descriptor,
        argv0=argv0,
        executable_path=executable_path,
        sha256=expected_sha256,
        identity=identity,
    )


def retain_control_plane_interpreter(
    python_executable: str,
) -> RetainedControlPlaneInterpreter:
    """Open, hash, and retain the exact root-owned Python executable."""

    with hashing_lock(kind="trusted-executable", exclusive=True):
        return _retain_control_plane_interpreter_unlocked(python_executable)


def _retain_control_plane_interpreter_unlocked(
    python_executable: str,
) -> RetainedControlPlaneInterpreter:
    executable = Path(python_executable).resolve(strict=True)
    lexical = os.lstat(executable)
    descriptor = os.open(
        executable,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
        current = os.lstat(executable)
        identity = _control_plane_interpreter_identity(before)
        if (
            _control_plane_interpreter_identity(lexical) != identity
            or _control_plane_interpreter_identity(after) != identity
            or _control_plane_interpreter_identity(current) != identity
        ):
            raise ValueError("sealed control-plane Python executable changed")
        return admit_retained_control_plane_interpreter(
            descriptor=descriptor,
            argv0=str(executable),
            expected_sha256="sha256:" + digest.hexdigest(),
        )
    except BaseException:
        os.close(descriptor)
        raise


def trusted_system_dependency_directories_json() -> str:
    """Attest only fixed root-owned system package directories, without `.pth`."""

    candidates = (
        Path(
            f"/usr/local/lib/python{sys.version_info.major}."
            f"{sys.version_info.minor}/dist-packages"
        ),
        Path("/usr/lib/python3/dist-packages"),
    )
    records: list[dict[str, object]] = []
    for path in candidates:
        if not path.exists():
            continue
        before = os.lstat(path)
        resolved = path.resolve(strict=True)
        after = os.lstat(path)
        identity = lambda value: (
            int(value.st_dev), int(value.st_ino), int(value.st_mode),
            int(value.st_uid), int(value.st_nlink), int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )
        if (
            resolved != path
            or identity(before) != identity(after)
            or not stat.S_ISDIR(before.st_mode)
            or before.st_uid != 0
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise ValueError("system dependency directory is not trusted")
        records.append(
            {
                "path": str(path),
                "st_dev": int(before.st_dev),
                "st_ino": int(before.st_ino),
                "st_mode": int(before.st_mode),
                "st_uid": int(before.st_uid),
                "st_nlink": int(before.st_nlink),
                "st_mtime_ns": int(before.st_mtime_ns),
                "st_ctime_ns": int(before.st_ctime_ns),
            }
        )
    if not records or not any(
        item["path"] == "/usr/lib/python3/dist-packages" for item in records
    ):
        raise ValueError("required system dependency directory is unavailable")
    return json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def admit_trusted_system_dependency_directories(value: str) -> tuple[str, ...]:
    """Recheck an exact directory attestation without executing site hooks."""

    expected = trusted_system_dependency_directories_json()
    if not isinstance(value, str) or value != expected:
        raise ValueError("system dependency directory attestation drifted")
    payload = json.loads(value)
    return tuple(str(item["path"]) for item in payload)


def sealed_native_dependency_environment(
    launch: AgentSupervisorNativeDependencyLaunch,
    *,
    system_dependency_directories_json: str,
) -> dict[str, str]:
    verify_agent_supervisor_native_dependency_sealed_fd(launch)
    admit_trusted_system_dependency_directories(
        system_dependency_directories_json
    )
    return {
        SEALED_NATIVE_DEPENDENCY_FD_ENV: str(launch.descriptor.descriptor),
        SEALED_NATIVE_DEPENDENCY_LAUNCH_ENV: launch.to_json(),
        SEALED_SYSTEM_DEPENDENCY_DIRS_ENV: system_dependency_directories_json,
    }


def admit_sealed_native_dependency_environment(
    environment: Mapping[str, str],
) -> tuple[AgentSupervisorNativeDependencyLaunch, str]:
    try:
        descriptor = int(environment[SEALED_NATIVE_DEPENDENCY_FD_ENV])
        text_value = environment[SEALED_NATIVE_DEPENDENCY_LAUNCH_ENV]
        system_value = environment[SEALED_SYSTEM_DEPENDENCY_DIRS_ENV]
        payload = json.loads(text_value, object_pairs_hook=_reject_duplicate_json_keys)
        launch = parse_agent_supervisor_native_dependency_launch(payload)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("sealed native dependency environment is invalid") from exc
    if (
        launch.to_json() != text_value
        or launch.descriptor.descriptor != descriptor
        or verify_agent_supervisor_native_dependency_sealed_fd(launch)
        != f"/proc/self/fd/{descriptor}"
    ):
        raise ValueError("sealed native dependency environment drifted")
    if launch != active_agent_supervisor_native_dependency_launch():
        raise ValueError("sealed native dependency is not the active launch")
    admit_trusted_system_dependency_directories(system_value)
    return launch, system_value


def optional_active_sealed_native_dependency(
    environment: Mapping[str, str],
) -> tuple[AgentSupervisorNativeDependencyLaunch, str] | None:
    """Return the active sealed DuckDB launch when the parent forwarded it.

    Absence of every native-launch field is a no-op so hermetic tests that
    never admitted DuckDB keep their current birth.  A partial envelope is
    still fail-closed through ``admit_sealed_native_dependency_environment``.
    """

    fd_text = str(environment.get(SEALED_NATIVE_DEPENDENCY_FD_ENV, "") or "").strip()
    launch_text = str(
        environment.get(SEALED_NATIVE_DEPENDENCY_LAUNCH_ENV, "") or ""
    ).strip()
    dirs_text = str(
        environment.get(SEALED_SYSTEM_DEPENDENCY_DIRS_ENV, "") or ""
    ).strip()
    if not fd_text and not launch_text and not dirs_text:
        return None
    return admit_sealed_native_dependency_environment(environment)


def preload_sealed_native_dependency_from_environment(
    environment: Mapping[str, str] | None = None,
) -> object | None:
    """Load the sealed DuckDB public alias before a cold supervisor import."""

    source = os.environ if environment is None else environment
    fd_text = str(source.get(SEALED_NATIVE_DEPENDENCY_FD_ENV, "") or "").strip()
    launch_text = str(
        source.get(SEALED_NATIVE_DEPENDENCY_LAUNCH_ENV, "") or ""
    ).strip()
    if not fd_text and not launch_text:
        return None
    from ipfs_accelerate_py.agent_implementation_route import (
        preload_agent_supervisor_native_dependency_from_bootstrap,
    )

    return preload_agent_supervisor_native_dependency_from_bootstrap(
        fd_text,
        launch_text,
    )


def apply_sealed_native_dependency_to_child_environment(
    environment: MutableMapping[str, str],
    *,
    parent_environment: Mapping[str, str] | None = None,
) -> tuple[int, ...]:
    """Copy an already-preloaded native launch into a child env and pass_fds."""

    source = os.environ if parent_environment is None else parent_environment
    admitted = optional_active_sealed_native_dependency(source)
    if admitted is None:
        return ()
    launch, directories = admitted
    environment.update(
        sealed_native_dependency_environment(
            launch,
            system_dependency_directories_json=directories,
        )
    )
    return (launch.descriptor.descriptor,)


def _python_executable_sha256(python_executable: str) -> tuple[str, str]:
    retained = retain_control_plane_interpreter(python_executable)
    try:
        return retained.argv0, retained.sha256
    finally:
        os.close(retained.descriptor)


def build_sealed_control_plane_module_command(
    *,
    python_executable: str,
    pin: AgentImplementationControlPlanePin,
    descriptor: int,
    module_name: str,
    argv: Sequence[str],
    retained_interpreter: RetainedControlPlaneInterpreter | None = None,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch | None = None,
    accepted_native_authorization_id: str = "",
    system_dependency_directories_json: str | None = None,
) -> list[str]:
    """Build one isolated, sealed-fd module launch with self-verifying bytes."""

    if module_name not in SEALED_CONTROL_PLANE_MODULES:
        raise ValueError("sealed control-plane target module is not allowed")
    verified_path = verify_agent_implementation_sealed_control_plane(
        pin,
        descriptor,
    )
    if verified_path != f"/proc/self/fd/{descriptor}":
        raise ValueError("sealed control-plane descriptor path drifted")
    if retained_interpreter is None:
        executable, executable_sha256 = _python_executable_sha256(
            python_executable
        )
    else:
        retained_interpreter = admit_retained_control_plane_interpreter(
            descriptor=retained_interpreter.descriptor,
            argv0=retained_interpreter.argv0,
            expected_sha256=retained_interpreter.sha256,
        )
        executable = retained_interpreter.argv0
        executable_sha256 = retained_interpreter.sha256
    if native_dependency_launch is None:
        raise ValueError("sealed control-plane native dependency is required")
    native_path = verify_agent_supervisor_native_dependency_sealed_fd(
        native_dependency_launch
    )
    native_descriptor = native_dependency_launch.descriptor.descriptor
    if (
        native_path != f"/proc/self/fd/{native_descriptor}"
        or native_dependency_launch.pin.python_executable_sha256
        != executable_sha256
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", accepted_native_authorization_id
        )
        is None
        or native_dependency_launch.accepted_authorization_id
        != accepted_native_authorization_id
    ):
        raise ValueError("sealed control-plane native dependency drifted")
    system_directories_json = (
        system_dependency_directories_json
        if system_dependency_directories_json is not None
        else trusted_system_dependency_directories_json()
    )
    admit_trusted_system_dependency_directories(system_directories_json)
    return [
        executable,
        "-I",
        "-S",
        "-c",
        SEALED_CONTROL_PLANE_BOOTSTRAP,
        str(descriptor),
        accepted_control_plane_pin_json(pin),
        accepted_native_authorization_id,
        str(native_descriptor),
        native_dependency_launch.to_json(),
        system_directories_json,
        module_name,
        SEALED_CONTROL_PLANE_BOOTSTRAP_SHA256,
        executable_sha256,
        *[str(item) for item in argv],
    ]


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return int(default)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


@dataclass(frozen=True)
class SupervisorTrack:
    """One supervisor process managed by the multi-supervisor runner."""

    name: str
    script_path: Path
    log_path: Path
    supervisor_pid_path: Path
    daemon_pid_path: Path
    supervisor_status_path: Path | None = None
    extra_args: tuple[str, ...] = ()
    module_name: str = ""
    database_program: DatabaseProgramConfig | None = None

    def resolve(self, repo_root: Path) -> SupervisorTrack:
        return SupervisorTrack(
            name=self.name,
            script_path=_resolve_path(repo_root, self.script_path),
            log_path=_resolve_path(repo_root, self.log_path),
            supervisor_pid_path=_resolve_path(repo_root, self.supervisor_pid_path),
            daemon_pid_path=_resolve_path(repo_root, self.daemon_pid_path),
            supervisor_status_path=(
                _resolve_path(repo_root, self.supervisor_status_path)
                if self.supervisor_status_path is not None
                else None
            ),
            extra_args=self.extra_args,
            module_name=self.module_name,
            database_program=self.database_program,
        )


@dataclass(frozen=True)
class _ManagedDaemonKernelFence:
    """Trusted parent projection for one lane's untrusted daemon markers.

    The sidecar remains same-UID writable and is therefore never authority by
    itself.  Signalling additionally requires an exact live birth in the
    immutable outer lane session, a dedicated daemon process group, the exact
    tracked owner scope, and exact procfs argv.
    """

    pid_path: Path
    identity_path: Path
    owner_scope: Mapping[str, str] | None
    root_session_id: int
    state_dir_option: str
    state_prefix: str
    todo_path_option: str | None
    daemon_entrypoint: str


# ---------------------------------------------------------------------------
# DatabaseProgramConfig@1 / DatabaseImplementationTrack@1
# ---------------------------------------------------------------------------
# Propagates explicit DuckDB/Quack authority selections from configured-board
# through multi-runner, implementation supervisor, and managed daemon without
# silent fallback to local file authority. Secret handles stay opaque; raw
# state credentials never enter provider subprocess environments.

DATABASE_PROGRAM_CONFIG_INTERFACE = "DatabaseProgramConfig@1"
DATABASE_IMPLEMENTATION_TRACK_INTERFACE = "DatabaseImplementationTrack@1"
DATABASE_PROGRAM_CONFIG_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/database-program-config@1"
)
DATABASE_IMPLEMENTATION_TRACK_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/database-implementation-track@1"
)

AUTHORITY_MODE_QUACK = "quack"
AUTHORITY_MODE_EMBEDDED = "embedded"
AUTHORITY_MODE_EMBEDDED_EXCLUSIVE = "embedded_exclusive"
AUTHORITY_MODE_LEGACY_MARKDOWN = "legacy_markdown"
CLOSED_AUTHORITY_MODES = frozenset(
    {
        AUTHORITY_MODE_QUACK,
        AUTHORITY_MODE_EMBEDDED,
        AUTHORITY_MODE_EMBEDDED_EXCLUSIVE,
        AUTHORITY_MODE_LEGACY_MARKDOWN,
    }
)

TASK_SOURCE_LEGACY_MARKDOWN = "legacy-markdown"
TASK_SOURCE_MARKDOWN = "markdown"
TASK_SOURCE_DUCKDB = "duckdb"
CLOSED_TASK_SOURCE_KINDS = frozenset(
    {
        TASK_SOURCE_LEGACY_MARKDOWN,
        TASK_SOURCE_MARKDOWN,
        TASK_SOURCE_DUCKDB,
    }
)

FAILOVER_FAIL_CLOSED = "fail_closed"
FAILOVER_REQUIRE_EXPLICIT = "require_explicit_operator"
CLOSED_FAILOVER_POLICIES = frozenset(
    {
        FAILOVER_FAIL_CLOSED,
        FAILOVER_REQUIRE_EXPLICIT,
    }
)

# Silent automatic fallbacks that would demote Quack authority.
FORBIDDEN_QUACK_FAILOVER_TARGETS = frozenset(
    {
        AUTHORITY_MODE_EMBEDDED,
        AUTHORITY_MODE_EMBEDDED_EXCLUSIVE,
        AUTHORITY_MODE_LEGACY_MARKDOWN,
        "file",
        "local_duckdb",
        "local-file",
        "markdown",
        "legacy-markdown",
    }
)

STATE_AUTHORITY_MODE_ENV = "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE"
STATE_QUACK_ENDPOINT_ENV = "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT"
STATE_QUACK_MUTATION_DIR_ENV = "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"
STATE_ENDPOINT_SECRET_HANDLE_ENV = (
    "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"
)
STATE_STORE_ID_ENV = "IPFS_ACCELERATE_AGENT_STATE_STORE_ID"
STATE_STORE_GENERATION_ENV = "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"
STATE_SCHEMA_REVISION_ENV = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
STATE_STORE_LIVE_GENERATION_ENV = (
    "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION"
)
STATE_LIVE_SCHEMA_REVISION_ENV = (
    "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION"
)
STATE_OWNER_SOCKET_ENV = "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET"
TASK_SOURCE_KIND_ENV = "IPFS_ACCELERATE_AGENT_TASK_SOURCE_KIND"
EVENT_STORE_PATH_ENV = "IPFS_ACCELERATE_AGENT_EVENT_STORE_PATH"
RUNTIME_REGISTRY_PATH_ENV = "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH"
EXPORT_PROFILE_ENV = "IPFS_ACCELERATE_AGENT_EXPORT_PROFILE"
STATE_FAILOVER_POLICY_ENV = "IPFS_ACCELERATE_AGENT_STATE_FAILOVER_POLICY"
DATABASE_PROGRAM_JSON_ENV = "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON"
TRUSTED_DUCKDB_HOME_ENV = "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME"
TRUSTED_PYTHON_USER_BASE_ENV = "PYTHONUSERBASE"
TRUSTED_XDG_CACHE_HOME_ENV = "XDG_CACHE_HOME"
TRUSTED_CUDA_CACHE_PATH_ENV = "CUDA_CACHE_PATH"
TRUSTED_CUDA_CACHE_DISABLE_ENV = "CUDA_CACHE_DISABLE"
TRUSTED_PYTHONDONTWRITEBYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"
TRUSTED_RUNTIME_CACHE_ENV_NAMES: tuple[str, ...] = (
    TRUSTED_XDG_CACHE_HOME_ENV,
    TRUSTED_CUDA_CACHE_PATH_ENV,
    TRUSTED_CUDA_CACHE_DISABLE_ENV,
    TRUSTED_PYTHONDONTWRITEBYTECODE_ENV,
)
STATE_GRANT_BROKER_SOCKET_ENV = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET"
)
STATE_GRANT_BROKER_SECRET_FD_ENV = (
    "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD"
)
TRUSTED_STATE_GRANT_BROKER_ENV_NAMES: tuple[str, ...] = (
    STATE_GRANT_BROKER_SOCKET_ENV,
    STATE_GRANT_BROKER_SECRET_FD_ENV,
)

DATABASE_PROGRAM_ENV_NAMES: tuple[str, ...] = (
    STATE_AUTHORITY_MODE_ENV,
    STATE_QUACK_ENDPOINT_ENV,
    STATE_QUACK_MUTATION_DIR_ENV,
    STATE_ENDPOINT_SECRET_HANDLE_ENV,
    STATE_STORE_ID_ENV,
    STATE_STORE_GENERATION_ENV,
    STATE_SCHEMA_REVISION_ENV,
    STATE_STORE_LIVE_GENERATION_ENV,
    STATE_LIVE_SCHEMA_REVISION_ENV,
    STATE_OWNER_SOCKET_ENV,
    TASK_SOURCE_KIND_ENV,
    EVENT_STORE_PATH_ENV,
    RUNTIME_REGISTRY_PATH_ENV,
    EXPORT_PROFILE_ENV,
    STATE_FAILOVER_POLICY_ENV,
    DATABASE_PROGRAM_JSON_ENV,
)

HASH_RESOURCE_ENV_NAMES: tuple[str, ...] = (
    "IPFS_HASH_CACHE_TTL_SECONDS",
    "IPFS_HASH_MAX_WORKERS",
    "IPFS_HASH_LOCK_TIMEOUT_SECONDS",
)

_PLAN_BOUND_PROFILE_ENV_NAMES = frozenset(
    {
        *ORDERED_IMPLEMENTATION_PROVIDER_ROUTE,
        *_ROUTE_AUTHORIZATION_ENV_NAMES,
        *_PROVIDER_EXECUTABLE_ENV_NAMES,
        *DATABASE_PROGRAM_ENV_NAMES,
        *HASH_RESOURCE_ENV_NAMES,
        PROVIDER_EXTERNAL_ISOLATION_ENV,
        TRUSTED_DUCKDB_HOME_ENV,
    }
)
_PLAN_BOUND_LIFECYCLE_ENV_NAMES = frozenset(
    {
        RUN_ID_ENV,
        PROFILE_ID_ENV,
        TARGET_ID_ENV,
        REPOSITORY_ROOT_ENV,
        STATE_ROOT_ENV,
        RUN_ROOT_ENV,
        FENCING_EPOCH_ENV,
        CONFIGURATION_ROOT_ENV,
    }
)


def _plan_bound_positive_child_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Project only sealed lane-control bindings into an accepted child."""

    allowed_names = {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        *_PLAN_BOUND_LIFECYCLE_ENV_NAMES,
        *_PLAN_BOUND_PROFILE_ENV_NAMES,
        *TRUSTED_STATE_GRANT_BROKER_ENV_NAMES,
    }
    trusted_home = str(environment.get(TRUSTED_DUCKDB_HOME_ENV, "") or "")
    trusted_runtime = (
        _trusted_duckdb_runtime_environment(
            environment,
            repository_root=Path(
                str(environment.get(REPOSITORY_ROOT_ENV, "") or "")
            ),
        )
        if trusted_home
        else {}
    )
    projected = {
        name: str(value)
        for name, value in environment.items()
        if name in allowed_names
    }
    if trusted_runtime:
        projected.update(trusted_runtime)
    else:
        projected.pop("HOME", None)
        projected.pop(TRUSTED_PYTHON_USER_BASE_ENV, None)
        for name in TRUSTED_RUNTIME_CACHE_ENV_NAMES:
            projected.pop(name, None)
    for name in tuple(projected):
        if (
            name.startswith(("PYTHON", "PYTEST", "LD_", "DYLD_"))
            or name == "GLIBC_TUNABLES"
        ):
            projected.pop(name, None)
    projected["PATH"] = "/usr/bin:/bin"
    return projected


def _plan_bound_profile_environment(
    environment: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Bind every positive non-lifecycle lane value into a profile CID."""

    return tuple(
        sorted(
            (name, str(environment[name]))
            for name in _PLAN_BOUND_PROFILE_ENV_NAMES
            if name in environment
        )
    )


def _validate_trusted_duckdb_home(
    value: str,
    *,
    repository_root: str,
    observed_home: str,
) -> Path:
    """Check the shape of a launcher-created DuckDB extension HOME binding."""

    if (
        not value
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
        or value != observed_home
    ):
        raise ValueError("trusted DuckDB HOME binding is incomplete")
    home = Path(value)
    root = Path(repository_root)
    if (
        not home.is_absolute()
        or not root.is_absolute()
        or home.parent.name != "qualification-homes"
        or re.fullmatch(r"[0-9a-f]{64}", home.name) is None
    ):
        raise ValueError("trusted DuckDB HOME binding is not canonical")
    try:
        home.relative_to(root)
        resolved_home = home.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
        resolved_home.relative_to(resolved_root)
        observed = os.lstat(home)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("trusted DuckDB HOME escapes the accepted repository") from exc
    if (
        resolved_home != home
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o500
    ):
        raise ValueError("trusted DuckDB HOME is not an immutable owned directory")
    return home


def _trusted_duckdb_runtime_environment(
    environment: Mapping[str, str],
    *,
    repository_root: Path,
) -> dict[str, str]:
    """Derive closed trusted-runtime bindings; never admit ambient cache paths."""

    trusted_home = str(environment.get(TRUSTED_DUCKDB_HOME_ENV, "") or "")
    python_user_base = str(environment.get(TRUSTED_PYTHON_USER_BASE_ENV, "") or "")
    if not trusted_home:
        raise ValueError("trusted DuckDB HOME binding is absent")
    home = _validate_trusted_duckdb_home(
        trusted_home,
        repository_root=str(repository_root.resolve()),
        observed_home=str(environment.get("HOME", "") or ""),
    )
    user_base: Path | None = None
    if python_user_base:
        user_base = Path(python_user_base)
        if (
            "\x00" in python_user_base
            or len(python_user_base.encode("utf-8")) > 4096
            or not user_base.is_absolute()
        ):
            raise ValueError("trusted Python user base binding is incomplete")
        try:
            user_base_observed = os.lstat(user_base)
            user_base_resolved = user_base.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("trusted Python user base binding is unavailable") from exc
        if (
            user_base_resolved != user_base
            or not stat.S_ISDIR(user_base_observed.st_mode)
            or stat.S_ISLNK(user_base_observed.st_mode)
            or user_base_observed.st_uid != os.geteuid()
            or stat.S_IMODE(user_base_observed.st_mode) & 0o022
        ):
            raise ValueError("trusted Python user base binding is unsafe")
    cache_root = home / ".cache"
    xdg_cache = cache_root / "xdg"
    cuda_cache = cache_root / "cuda"
    for directory in (cache_root, xdg_cache, cuda_cache):
        try:
            observed = os.lstat(directory)
            resolved = directory.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("trusted runtime cache directory is unavailable") from exc
        if (
            resolved != directory
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise ValueError("trusted runtime cache directory is unsafe")
    result = {
        "HOME": str(home),
        TRUSTED_DUCKDB_HOME_ENV: str(home),
        TRUSTED_XDG_CACHE_HOME_ENV: str(xdg_cache),
        TRUSTED_CUDA_CACHE_PATH_ENV: str(cuda_cache),
        TRUSTED_CUDA_CACHE_DISABLE_ENV: "1",
        TRUSTED_PYTHONDONTWRITEBYTECODE_ENV: "1",
    }
    if user_base is not None:
        result[TRUSTED_PYTHON_USER_BASE_ENV] = str(user_base)
    return result


def _trusted_duckdb_profile_environment(
    environment: Mapping[str, str],
    *,
    repository_root: Path,
) -> tuple[tuple[str, str], ...]:
    """Bind an admitted extension HOME into a lifecycle profile."""

    if not str(environment.get(TRUSTED_DUCKDB_HOME_ENV, "") or ""):
        return ()
    return tuple(
        sorted(
            _trusted_duckdb_runtime_environment(
                environment,
                repository_root=repository_root,
            ).items()
        )
    )

# Raw state credentials that must never reach implementation-provider children.
STATE_CREDENTIAL_ENV_NAMES: frozenset[str] = frozenset(
    {
        "QUACK_TOKEN",
        "QUACK_PASSWORD",
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        STATE_QUACK_MUTATION_DIR_ENV,
        RUNTIME_REGISTRY_PATH_ENV,
        "QUACK_SECRET",
        "DUCKDB_TOKEN",
        "DUCKDB_PASSWORD",
        "DUCKDB_SECRET",
        "IPFS_ACCELERATE_AGENT_QUACK_PASSWORD",
        "IPFS_ACCELERATE_AGENT_QUACK_SECRET",
        "IPFS_ACCELERATE_AGENT_STATE_TOKEN",
        "IPFS_ACCELERATE_AGENT_STATE_PASSWORD",
        "IPFS_ACCELERATE_AGENT_STATE_SECRET",
        "IPFS_ACCELERATE_AGENT_STATE_CREDENTIAL",
        *TRUSTED_STATE_GRANT_BROKER_ENV_NAMES,
        "IPFS_ACCELERATE_AGENT_CONTROL_PLANE_TOKEN",
        "IPFS_ACCELERATE_AGENT_CONTROL_PLANE_PASSWORD",
    }
)

_SECRET_HANDLE_PREFIXES = (
    "env://",
    "vault://",
    "handle:",
    "secret-handle:",
)
_REDACTION_MARKER = "secret_material"
_SAFE_HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,255}$")


class DatabaseProgramConfigError(ValueError):
    """Raised when a database program selection is missing, unsafe, or incomplete."""


class _SupportsFileno(Protocol):
    def fileno(self) -> int: ...


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return int(default)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _is_secret_handle(value: str) -> bool:
    text = value.strip()
    return any(text.startswith(prefix) for prefix in _SECRET_HANDLE_PREFIXES)


def _require_nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatabaseProgramConfigError(f"{field} must be a nonempty string")
    text = value.strip()
    if "\x00" in text or "\n" in text or "\r" in text:
        raise DatabaseProgramConfigError(f"{field} must be a single-line string")
    return text


def _require_safe_id(value: Any, *, field: str) -> str:
    text = _require_nonempty_text(value, field=field)
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise DatabaseProgramConfigError(f"{field} is not a safe identifier")
    return text


def _require_secret_handle(value: Any, *, field: str) -> str:
    text = _require_nonempty_text(value, field=field)
    if not _is_secret_handle(text):
        raise DatabaseProgramConfigError(
            f"{field} must be an opaque secret handle "
            f"(env://, vault://, handle:, or secret-handle:); "
            "raw credentials are forbidden"
        )
    if _SAFE_HANDLE_RE.fullmatch(text) is None:
        raise DatabaseProgramConfigError(f"{field} is not a safe secret handle")
    return text


def _optional_relative_path(value: Any, *, field: str) -> str:
    if value is None or value == "":
        return ""
    text = _require_nonempty_text(value, field=field)
    path = Path(text)
    if (
        text in {".", ".."}
        or ".." in path.parts
        or "://" in text
        # LGSWF-062: hermetic tests pass absolute tmp worktree roots.
        or re.match(r"^[A-Za-z]:[\\/]", text)
    ):
        raise DatabaseProgramConfigError(
            f"{field} must be a safe repository-relative path"
        )
    return path.as_posix()


def _optional_worktree_root(value: Any, *, field: str) -> str:
    """Accept relative or absolute worktree roots from production CLI argv.

    Event/registry paths remain repository-relative, but ``--worktree-root`` is
    commonly an absolute path under the board repo (tests and managed daemons).
    Reject parent-escape segments and URL schemes only.
    """

    if value is None or value == "":
        return ""
    text = _require_nonempty_text(value, field=field)
    path = Path(text)
    if text in {".", ".."} or ".." in path.parts or "://" in text:
        raise DatabaseProgramConfigError(
            f"{field} must be a safe worktree path without parent escape"
        )
    return path.as_posix()


@dataclass(frozen=True)
class DatabaseProgramConfig:
    """Explicit database/Quack authority selection for one program (DatabaseProgramConfig@1)."""

    INTERFACE: ClassVar[str] = DATABASE_PROGRAM_CONFIG_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_PROGRAM_CONFIG_SCHEMA

    authority_mode: str
    task_source_kind: str
    endpoint_secret_handle: str = ""
    quack_endpoint: str = ""
    store_id: str = ""
    store_generation: str = ""
    schema_revision: str = ""
    event_store_path: str = ""
    runtime_registry_path: str = ""
    worktree_root: str = ""
    export_profile: str = ""
    failover_policy: str = FAILOVER_FAIL_CLOSED
    explicit_legacy: bool = False

    def __post_init__(self) -> None:
        mode = str(self.authority_mode or "").strip().lower().replace("-", "_")
        if mode not in CLOSED_AUTHORITY_MODES:
            raise DatabaseProgramConfigError(
                f"unsupported authority_mode: {self.authority_mode!r}"
            )
        object.__setattr__(self, "authority_mode", mode)

        kind = str(self.task_source_kind or "").strip().lower()
        if kind not in CLOSED_TASK_SOURCE_KINDS:
            raise DatabaseProgramConfigError(
                f"unsupported task_source_kind: {self.task_source_kind!r}"
            )
        object.__setattr__(self, "task_source_kind", kind)

        failover = str(self.failover_policy or FAILOVER_FAIL_CLOSED).strip().lower()
        if failover not in CLOSED_FAILOVER_POLICIES:
            raise DatabaseProgramConfigError(
                f"unsupported failover_policy: {self.failover_policy!r}"
            )
        object.__setattr__(self, "failover_policy", failover)

        handle = str(self.endpoint_secret_handle or "").strip()
        if handle:
            handle = _require_secret_handle(
                handle,
                field="endpoint_secret_handle",
            )
        object.__setattr__(self, "endpoint_secret_handle", handle)

        quack_endpoint = str(self.quack_endpoint or "").strip()
        if quack_endpoint:
            from ..task_sources.duckdb_state import is_quack_transport_target

            if not is_quack_transport_target(quack_endpoint):
                raise DatabaseProgramConfigError(
                    "quack_endpoint must be a loopback quack: URI"
                )
        object.__setattr__(self, "quack_endpoint", quack_endpoint)

        store_id = str(self.store_id or "").strip()
        if store_id:
            store_id = _require_safe_id(store_id, field="store_id")
        object.__setattr__(self, "store_id", store_id)

        generation = str(self.store_generation or "").strip()
        if generation and _SAFE_ID_RE.fullmatch(generation) is None:
            raise DatabaseProgramConfigError(
                "store_generation is not a safe generation token"
            )
        object.__setattr__(self, "store_generation", generation)

        schema = str(self.schema_revision or "").strip()
        if schema and _SAFE_ID_RE.fullmatch(schema) is None:
            raise DatabaseProgramConfigError(
                "schema_revision is not a safe schema identifier"
            )
        object.__setattr__(self, "schema_revision", schema)

        object.__setattr__(
            self,
            "event_store_path",
            _optional_relative_path(
                self.event_store_path,
                field="event_store_path",
            ),
        )
        object.__setattr__(
            self,
            "runtime_registry_path",
            _optional_relative_path(
                self.runtime_registry_path,
                field="runtime_registry_path",
            ),
        )
        object.__setattr__(
            self,
            "worktree_root",
            _optional_worktree_root(
                self.worktree_root,
                field="worktree_root",
            ),
        )

        export_profile = str(self.export_profile or "").strip()
        if export_profile and _SAFE_ID_RE.fullmatch(export_profile) is None:
            raise DatabaseProgramConfigError(
                "export_profile is not a safe profile identifier"
            )
        object.__setattr__(self, "export_profile", export_profile)
        object.__setattr__(self, "explicit_legacy", bool(self.explicit_legacy))

        if mode == AUTHORITY_MODE_LEGACY_MARKDOWN:
            if kind not in {
                TASK_SOURCE_LEGACY_MARKDOWN,
                TASK_SOURCE_MARKDOWN,
            }:
                raise DatabaseProgramConfigError(
                    "legacy_markdown authority requires task_source_kind "
                    "'legacy-markdown' or 'markdown'"
                )
            if not self.explicit_legacy:
                raise DatabaseProgramConfigError(
                    "legacy_markdown authority requires explicit_legacy=true; "
                    "the implicit legacy-Markdown default is deprecated"
                )
        if mode == AUTHORITY_MODE_QUACK:
            if not handle:
                raise DatabaseProgramConfigError(
                    "quack authority requires endpoint_secret_handle"
                )
            if not quack_endpoint:
                raise DatabaseProgramConfigError(
                    "quack authority requires quack_endpoint"
                )
            if not store_id:
                raise DatabaseProgramConfigError(
                    "quack authority requires store_id"
                )
            if not generation:
                raise DatabaseProgramConfigError(
                    "quack authority requires store_generation"
                )
            if not schema:
                raise DatabaseProgramConfigError(
                    "quack authority requires schema_revision"
                )
            if kind == TASK_SOURCE_LEGACY_MARKDOWN:
                raise DatabaseProgramConfigError(
                    "quack authority cannot use legacy-markdown task source"
                )
            if failover != FAILOVER_FAIL_CLOSED:
                # Quack may only fail closed; never silently become local
                # DuckDB or file authority under any other policy.
                raise DatabaseProgramConfigError(
                    "quack authority requires failover_policy='fail_closed'; "
                    "silent local DuckDB/file fallback is forbidden"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DATABASE_PROGRAM_CONFIG_SCHEMA,
            "interface": DATABASE_PROGRAM_CONFIG_INTERFACE,
            "authority_mode": self.authority_mode,
            "task_source_kind": self.task_source_kind,
            "endpoint_secret_handle": self.endpoint_secret_handle,
            "quack_endpoint": self.quack_endpoint,
            "store_id": self.store_id,
            "store_generation": self.store_generation,
            "schema_revision": self.schema_revision,
            "event_store_path": self.event_store_path,
            "runtime_registry_path": self.runtime_registry_path,
            "worktree_root": self.worktree_root,
            "export_profile": self.export_profile,
            "failover_policy": self.failover_policy,
            "explicit_legacy": self.explicit_legacy,
        }

    def redacted_dict(self) -> dict[str, Any]:
        """Return a public projection that never exposes raw secret material."""

        payload = self.to_dict()
        if payload["endpoint_secret_handle"]:
            # Handles are opaque references and safe to publish; raw tokens
            # are rejected at parse time so this never contains credentials.
            payload["endpoint_secret_handle"] = payload["endpoint_secret_handle"]
        return payload

    def cli_args(self) -> list[str]:
        """Return supervisor/daemon CLI options that preserve this selection."""

        args = [
            "--task-source-kind",
            self.task_source_kind,
            "--authority-mode",
            self.authority_mode,
            "--state-failover-policy",
            self.failover_policy,
        ]
        if self.endpoint_secret_handle:
            args.extend(
                ["--endpoint-secret-handle", self.endpoint_secret_handle]
            )
        if self.quack_endpoint:
            args.extend(["--quack-endpoint", self.quack_endpoint])
        if self.store_id:
            args.extend(["--state-store-id", self.store_id])
        if self.store_generation:
            args.extend(["--state-store-generation", self.store_generation])
        if self.schema_revision:
            args.extend(["--state-schema-revision", self.schema_revision])
        if self.event_store_path:
            args.extend(["--event-store-path", self.event_store_path])
        if self.runtime_registry_path:
            args.extend(["--runtime-registry-path", self.runtime_registry_path])
        if self.worktree_root:
            # Prefer the existing supervisor worktree root flag when set.
            args.extend(["--worktree-root", self.worktree_root])
        if self.export_profile:
            args.extend(["--export-profile", self.export_profile])
        if self.explicit_legacy:
            args.append("--explicit-legacy-task-source")
        return args

    def environment(
        self,
        *,
        repository_root: str | Path | None = None,
    ) -> dict[str, str]:
        """Return non-secret environment bindings for child supervisors/daemons."""

        env = {
            STATE_AUTHORITY_MODE_ENV: self.authority_mode,
            TASK_SOURCE_KIND_ENV: self.task_source_kind,
            STATE_FAILOVER_POLICY_ENV: self.failover_policy,
            DATABASE_PROGRAM_JSON_ENV: json.dumps(
                self.to_dict(),
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        if self.endpoint_secret_handle:
            env[STATE_ENDPOINT_SECRET_HANDLE_ENV] = self.endpoint_secret_handle
        if self.quack_endpoint:
            env[STATE_QUACK_ENDPOINT_ENV] = self.quack_endpoint
        if self.store_id:
            env[STATE_STORE_ID_ENV] = self.store_id
        if self.store_generation:
            env[STATE_STORE_GENERATION_ENV] = self.store_generation
        if self.schema_revision:
            env[STATE_SCHEMA_REVISION_ENV] = self.schema_revision
        if self.event_store_path:
            env[EVENT_STORE_PATH_ENV] = self.event_store_path
        if self.authority_mode == AUTHORITY_MODE_QUACK:
            if not self.runtime_registry_path:
                raise DatabaseProgramConfigError(
                    "quack authority requires runtime_registry_path for "
                    "owner/worker mutation binding"
                )
            from ..task_sources.duckdb_state import (
                DuckDBConnectionPolicyError,
                quack_owner_mutation_inbox_path,
            )

            try:
                mutation_inbox = quack_owner_mutation_inbox_path(
                    self.runtime_registry_path,
                    repository_root=repository_root,
                )
            except DuckDBConnectionPolicyError as exc:
                raise DatabaseProgramConfigError(str(exc)) from exc
            env[RUNTIME_REGISTRY_PATH_ENV] = str(mutation_inbox.parent)
            env[STATE_QUACK_MUTATION_DIR_ENV] = str(mutation_inbox)
        elif self.runtime_registry_path:
            env[RUNTIME_REGISTRY_PATH_ENV] = self.runtime_registry_path
        if self.export_profile:
            env[EXPORT_PROFILE_ENV] = self.export_profile
        return env

    def daemon_cli_args(self) -> list[str]:
        """Return explicit non-secret authority options for a managed daemon.

        The environment carries the same immutable program as a defense in
        depth, but store selection must remain inspectable in the process
        command and must not depend on ambient environment propagation.  The
        endpoint secret *handle* remains environment-only; it is an opaque
        reference, but keeping it out of argv also satisfies configurations
        that prohibit all credential references in process listings.
        """

        args = [
            "--task-source-kind",
            self.task_source_kind,
            "--authority-mode",
            self.authority_mode,
            "--state-failover-policy",
            self.failover_policy,
        ]
        if self.quack_endpoint:
            args.extend(["--quack-endpoint", self.quack_endpoint])
        if self.store_id:
            args.extend(["--state-store-id", self.store_id])
        if self.store_generation:
            args.extend(["--state-store-generation", self.store_generation])
        if self.schema_revision:
            args.extend(["--state-schema-revision", self.schema_revision])
        if self.event_store_path:
            args.extend(["--event-store-path", self.event_store_path])
        if self.runtime_registry_path:
            args.extend(["--runtime-registry-path", self.runtime_registry_path])
        if self.export_profile:
            args.extend(["--export-profile", self.export_profile])
        if self.explicit_legacy:
            args.append("--explicit-legacy-task-source")
        return args

    def assert_quack_not_demoted(self, *, candidate_mode: str) -> None:
        """Fail closed when a Quack selection would become local/file authority."""

        if self.authority_mode != AUTHORITY_MODE_QUACK:
            return
        target = str(candidate_mode or "").strip().lower().replace("-", "_")
        if target in FORBIDDEN_QUACK_FAILOVER_TARGETS or target != AUTHORITY_MODE_QUACK:
            raise DatabaseProgramConfigError(
                "quack authority cannot silently become local DuckDB or file "
                f"authority (attempted {candidate_mode!r})"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DatabaseProgramConfig:
        if not isinstance(payload, Mapping):
            raise DatabaseProgramConfigError(
                "database program config must be an object"
            )
        return cls(
            authority_mode=str(payload.get("authority_mode") or ""),
            task_source_kind=str(payload.get("task_source_kind") or ""),
            endpoint_secret_handle=str(
                payload.get("endpoint_secret_handle") or ""
            ),
            quack_endpoint=str(payload.get("quack_endpoint") or ""),
            store_id=str(payload.get("store_id") or ""),
            store_generation=str(
                payload.get("store_generation")
                if payload.get("store_generation") is not None
                else ""
            ),
            schema_revision=str(payload.get("schema_revision") or ""),
            event_store_path=str(payload.get("event_store_path") or ""),
            runtime_registry_path=str(
                payload.get("runtime_registry_path") or ""
            ),
            worktree_root=str(payload.get("worktree_root") or ""),
            export_profile=str(payload.get("export_profile") or ""),
            failover_policy=str(
                payload.get("failover_policy") or FAILOVER_FAIL_CLOSED
            ),
            explicit_legacy=bool(payload.get("explicit_legacy", False)),
        )

    @classmethod
    def explicit_legacy_markdown(cls) -> DatabaseProgramConfig:
        """Return an explicit legacy Markdown program selection (not implicit)."""

        return cls(
            authority_mode=AUTHORITY_MODE_LEGACY_MARKDOWN,
            task_source_kind=TASK_SOURCE_LEGACY_MARKDOWN,
            failover_policy=FAILOVER_FAIL_CLOSED,
            explicit_legacy=True,
        )


def parse_database_program_config(
    payload: Mapping[str, Any] | None,
) -> DatabaseProgramConfig | None:
    """Parse an optional database_program mapping; None when absent."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DatabaseProgramConfigError("database_program must be an object")
    if not payload:
        return None
    return DatabaseProgramConfig.from_mapping(payload)


def redact_database_program_argv(argv: Sequence[str]) -> list[str]:
    """Return argv with secret-bearing values replaced by a redaction marker.

    Opaque secret handles remain visible (they are references, not credentials).
    Values that look like raw tokens after known credential flags are redacted.
    """

    redacted: list[str] = []
    redact_next = False
    credential_flags = {
        "--state-token",
        "--quack-token",
        "--state-password",
        "--quack-password",
        "--state-secret",
        "--quack-secret",
    }
    for item in argv:
        token = str(item)
        if redact_next:
            redacted.append(_REDACTION_MARKER)
            redact_next = False
            continue
        if token in credential_flags:
            redacted.append(token)
            redact_next = True
            continue
        if "=" in token:
            name, value = token.split("=", 1)
            if name in credential_flags or (
                any(
                    needle in name.lower()
                    for needle in ("token", "password", "secret", "credential")
                )
                and not _is_secret_handle(value)
            ):
                redacted.append(f"{name}={_REDACTION_MARKER}")
                continue
        redacted.append(token)
    return redacted


def scrub_state_credentials_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    secret_handle: str = "",
) -> dict[str, str]:
    """Return a copy of ``environment`` without state credentials.

    Provider subprocesses must never inherit Quack/DuckDB tokens. Opaque
    secret handles (references) may remain; resolved env:// targets named by
    the handle are also removed.
    """

    source = dict(os.environ if environment is None else environment)
    cleaned = {
        key: value
        for key, value in source.items()
        if key not in STATE_CREDENTIAL_ENV_NAMES
        and not key.upper().endswith(("_QUACK_TOKEN", "_STATE_TOKEN", "_QUACK_SECRET"))
    }
    handle = str(secret_handle or "").strip()
    if handle.startswith("env://"):
        target = handle[len("env://") :].strip()
        if target:
            cleaned.pop(target, None)
    return cleaned


def provider_subprocess_environment(
    environment: Mapping[str, str] | None = None,
    *,
    program: DatabaseProgramConfig | None = None,
) -> dict[str, str]:
    """Environment safe for implementation-provider children."""

    handle = program.endpoint_secret_handle if program is not None else ""
    cleaned = scrub_state_credentials_from_environment(
        environment,
        secret_handle=handle,
    )
    from .process_security import (
        STATE_AUTHORITY_CREDENTIAL_NAMES,
        STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV,
        STATE_AUTHORITY_HANDOFF_ENV_NAMES,
    )

    for name in (
        *STATE_AUTHORITY_CREDENTIAL_NAMES,
        *STATE_AUTHORITY_HANDOFF_ENV_NAMES,
        STATE_AUTHORITY_DESCRIPTOR_SOCKET_ENV,
    ):
        cleaned.pop(name, None)
    # Provider children also must not receive the supervisor's state-authority
    # bindings; they operate on worktree files only.
    for name in DATABASE_PROGRAM_ENV_NAMES:
        cleaned.pop(name, None)
    for name in (
        SEALED_NATIVE_DEPENDENCY_FD_ENV,
        SEALED_NATIVE_DEPENDENCY_LAUNCH_ENV,
        SEALED_SYSTEM_DEPENDENCY_DIRS_ENV,
    ):
        cleaned.pop(name, None)
    # Provider children operate on worktree files only.  Leaking a subset of
    # the supervisor lifecycle identity makes grok_cli_runner fail-close on a
    # partial Docker cleanup binding and quarantines the claim with no effect.
    for name in _PLAN_BOUND_LIFECYCLE_ENV_NAMES:
        cleaned.pop(name, None)
    cleaned.pop(PROVIDER_EXTERNAL_ISOLATION_ENV, None)
    trusted_home = str(cleaned.pop(TRUSTED_DUCKDB_HOME_ENV, "") or "")
    cleaned.pop(TRUSTED_PYTHON_USER_BASE_ENV, None)
    for name in TRUSTED_RUNTIME_CACHE_ENV_NAMES:
        cleaned.pop(name, None)
    if trusted_home and cleaned.get("HOME") == trusted_home:
        cleaned.pop("HOME", None)
    return cleaned




@dataclass(frozen=True)
class ImplementationSupervisorTrackConfig:
    """Structured inputs for one implementation-supervisor track."""

    name: str
    script_path: Path | str
    state_dir: Path | str
    state_prefix: str
    database_program: DatabaseProgramConfig | None = None

    def compact_spec(self) -> str:
        """Return the compact CLI ``--implementation-track`` spec."""

        return implementation_supervisor_compact_track_spec(
            name=self.name,
            script_path=self.script_path,
            state_dir=self.state_dir,
            state_prefix=self.state_prefix,
        )

    def track_spec(self) -> str:
        """Return the expanded supervisor track spec with log and PID paths."""

        return implementation_supervisor_track_spec(
            name=self.name,
            script_path=self.script_path,
            state_dir=self.state_dir,
            state_prefix=self.state_prefix,
        )



@dataclass(frozen=True)
class DatabaseImplementationTrack:
    """Implementation track bound to an explicit database program selection."""

    INTERFACE: ClassVar[str] = DATABASE_IMPLEMENTATION_TRACK_INTERFACE
    SCHEMA: ClassVar[str] = DATABASE_IMPLEMENTATION_TRACK_SCHEMA

    name: str
    script_path: Path | str
    state_dir: Path | str
    state_prefix: str
    database_program: DatabaseProgramConfig
    lane_index: int | None = None
    lane_count: int | None = None

    def track_config(self) -> ImplementationSupervisorTrackConfig:
        return ImplementationSupervisorTrackConfig(
            name=self.name,
            script_path=self.script_path,
            state_dir=self.state_dir,
            state_prefix=self.state_prefix,
            database_program=self.database_program,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DATABASE_IMPLEMENTATION_TRACK_SCHEMA,
            "interface": DATABASE_IMPLEMENTATION_TRACK_INTERFACE,
            "name": self.name,
            "script_path": Path(self.script_path).as_posix(),
            "state_dir": Path(self.state_dir).as_posix(),
            "state_prefix": self.state_prefix,
            "database_program": self.database_program.to_dict(),
            "lane_index": self.lane_index,
            "lane_count": self.lane_count,
        }

    def common_args(self) -> list[str]:
        return self.database_program.cli_args()


@dataclass(frozen=True)
class PlanBoundSupervisorChild:
    """One exact nonempty CAS-bound supervisor slice.

    The JSON CLI record is a transport projection only.  The child reloads
    and verifies every identity against ``PlanRevisionStore`` before it may
    start the existing implementation daemon.
    """

    name: str
    accepted_tree_root: Path | str
    script_path: Path | str
    state_dir: Path | str
    state_prefix: str
    plan_revision_store_path: Path | str
    revision_cid: str
    plan_root_cid: str
    execution_plan_cid: str
    capacity_snapshot_id: str
    slice_manifest_cid: str
    slice_id: str
    source_head: str
    source_tree: str
    task_source_revision: str
    configuration_root: str
    lane_id: str
    task_ids: tuple[str, ...]
    task_cids: tuple[str, ...]
    reassignment_cid: str = ""

    def __post_init__(self) -> None:
        def relative_path(value: Path | str, field_name: str) -> str:
            if not isinstance(value, (Path, str)):
                raise ValueError(f"plan-bound child {field_name} must be a path")
            raw = str(value)
            normalized = raw.replace("\\", "/")
            parsed = PurePosixPath(normalized)
            if (
                raw != normalized
                or not normalized
                or parsed.is_absolute()
                or ".." in parsed.parts
                or parsed.as_posix() in {".", ".."}
                or parsed.as_posix() != normalized
            ):
                raise ValueError(
                    f"plan-bound child {field_name} must be a safe relative path"
                )
            return normalized

        if not isinstance(self.accepted_tree_root, (Path, str)):
            raise ValueError("plan-bound child accepted_tree_root must be a path")
        accepted_tree_root = Path(self.accepted_tree_root)
        if not accepted_tree_root.is_absolute():
            raise ValueError(
                "plan-bound child accepted_tree_root must be absolute"
            )
        accepted_tree_root = _canonical_accepted_tree_root(accepted_tree_root)
        object.__setattr__(self, "accepted_tree_root", str(accepted_tree_root))
        script_path = relative_path(self.script_path, "script_path")
        if script_path != PLAN_BOUND_ACCEPTED_ENTRY_PATH:
            raise ValueError("plan-bound child script_path is not the accepted entry")
        object.__setattr__(self, "script_path", script_path)
        state_dir = relative_path(self.state_dir, "state_dir")
        store_path = relative_path(
            self.plan_revision_store_path,
            "plan_revision_store_path",
        )
        state_parent = PurePosixPath(state_dir).parent
        store_parent = PurePosixPath(store_path).parent
        if (
            state_parent != store_parent
            or store_parent == PurePosixPath(".")
            or PurePosixPath(store_path).name != "plan-revision-store"
        ):
            raise ValueError(
                "plan-bound child state and plan store do not share authority root"
            )
        object.__setattr__(self, "state_dir", state_dir)
        object.__setattr__(self, "plan_revision_store_path", store_path)
        if len(self.task_ids) != 1 or len(self.task_cids) != 1:
            raise ValueError(
                "plan-bound children require one exact ID/CID task pair"
            )
        for field_name in (
            "name", "state_prefix", "revision_cid", "plan_root_cid",
            "execution_plan_cid", "capacity_snapshot_id", "slice_manifest_cid",
            "slice_id", "source_head", "source_tree", "task_source_revision",
            "configuration_root", "lane_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"plan-bound child {field_name} is required")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.name) is None:
            raise ValueError("plan-bound child name is unsafe")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.state_prefix) is None:
            raise ValueError("plan-bound child state_prefix is unsafe")
        for field_name in ("source_head", "source_tree"):
            if re.fullmatch(r"[0-9a-f]{40,64}", getattr(self, field_name)) is None:
                raise ValueError(
                    f"plan-bound child {field_name} is not a Git object identity"
                )
        for field_name in ("task_ids", "task_cids"):
            values = getattr(self, field_name)
            if (
                not isinstance(values, tuple)
                or any(
                    not isinstance(item, str)
                    or not item.strip()
                    or item != item.strip()
                    for item in values
                )
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"plan-bound child {field_name} must be exact unique strings"
                )
        if not isinstance(self.reassignment_cid, str):
            raise ValueError("plan-bound child reassignment_cid must be text")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PLAN_BOUND_CHILD_SCHEMA,
            "name": self.name,
            "accepted_tree_root": str(self.accepted_tree_root),
            "script_path": str(self.script_path),
            "state_dir": str(self.state_dir),
            "state_prefix": self.state_prefix,
            "plan_revision_store_path": str(self.plan_revision_store_path),
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "execution_plan_cid": self.execution_plan_cid,
            "capacity_snapshot_id": self.capacity_snapshot_id,
            "slice_manifest_cid": self.slice_manifest_cid,
            "slice_id": self.slice_id,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "task_source_revision": self.task_source_revision,
            "configuration_root": self.configuration_root,
            "lane_id": self.lane_id,
            "task_ids": list(self.task_ids),
            "task_cids": list(self.task_cids),
            "reassignment_cid": self.reassignment_cid,
        }

    def cli_record(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_cli_record(cls, value: str) -> PlanBoundSupervisorChild:
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(
                        f"plan-bound child record has duplicate key {key!r}"
                    )
                result[key] = item
            return result

        try:
            payload = json.loads(value, object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("plan-bound child record is invalid JSON") from exc
        expected_fields = {
            "schema", "name", "accepted_tree_root", "script_path",
            "state_dir", "state_prefix", "plan_revision_store_path",
            "revision_cid", "plan_root_cid", "execution_plan_cid",
            "capacity_snapshot_id", "slice_manifest_cid", "slice_id",
            "source_head", "source_tree", "task_source_revision",
            "configuration_root", "lane_id", "task_ids", "task_cids",
            "reassignment_cid",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected_fields:
            raise ValueError("plan-bound child record fields are not exact")
        if payload.get("schema") != PLAN_BOUND_CHILD_SCHEMA:
            raise ValueError("plan-bound child record has an unsupported schema")
        scalar_fields = expected_fields - {"task_ids", "task_cids"}
        if any(not isinstance(payload[name], str) for name in scalar_fields):
            raise ValueError("plan-bound child record text fields are invalid")
        if any(
            not isinstance(payload[name], list)
            or any(not isinstance(item, str) for item in payload[name])
            for name in ("task_ids", "task_cids")
        ):
            raise ValueError("plan-bound child record task populations are invalid")
        result = cls(
            name=payload["name"],
            accepted_tree_root=payload["accepted_tree_root"],
            script_path=payload["script_path"],
            state_dir=payload["state_dir"],
            state_prefix=payload["state_prefix"],
            plan_revision_store_path=payload["plan_revision_store_path"],
            revision_cid=payload["revision_cid"],
            plan_root_cid=payload["plan_root_cid"],
            execution_plan_cid=payload["execution_plan_cid"],
            capacity_snapshot_id=payload["capacity_snapshot_id"],
            slice_manifest_cid=payload["slice_manifest_cid"],
            slice_id=payload["slice_id"],
            source_head=payload["source_head"],
            source_tree=payload["source_tree"],
            task_source_revision=payload["task_source_revision"],
            configuration_root=payload["configuration_root"],
            lane_id=payload["lane_id"],
            task_ids=tuple(payload["task_ids"]),
            task_cids=tuple(payload["task_cids"]),
            reassignment_cid=payload["reassignment_cid"],
        )
        if payload != result.to_dict():
            raise ValueError("plan-bound child record changed during decoding")
        return result

    def track(self, *, stamp: str = "") -> SupervisorTrack:
        base = parse_implementation_track_spec(
            implementation_supervisor_compact_track_spec(
                name=self.name,
                script_path=self.script_path,
                state_dir=self.state_dir,
                state_prefix=self.state_prefix,
            ),
            stamp=stamp,
        )
        args = [
            *base.extra_args,
            "--plan-bound-dispatch",
            "--plan-revision-store-path", str(self.plan_revision_store_path),
            "--plan-bound-revision-cid", self.revision_cid,
            "--plan-bound-plan-root-cid", self.plan_root_cid,
            "--plan-bound-execution-plan-cid", self.execution_plan_cid,
            "--plan-bound-capacity-snapshot-id", self.capacity_snapshot_id,
            "--plan-bound-slice-manifest-cid", self.slice_manifest_cid,
            "--plan-bound-slice-id", self.slice_id,
            "--plan-bound-source-head", self.source_head,
            "--plan-bound-source-tree", self.source_tree,
            "--plan-bound-task-source-revision", self.task_source_revision,
            "--plan-bound-configuration-root", self.configuration_root,
            "--plan-bound-accepted-tree-root", str(self.accepted_tree_root),
            "--plan-bound-lane-id", self.lane_id,
            "--task-shard-count", "1",
            "--task-shard-index", "0",
        ]
        if self.reassignment_cid:
            args.extend(
                ["--plan-bound-reassignment-cid", self.reassignment_cid]
            )
        for task_id in self.task_ids:
            args.extend(("--execution-slice-task-id", task_id))
        for task_cid in self.task_cids:
            args.extend(("--execution-slice-task-cid", task_cid))
        return SupervisorTrack(
            name=base.name,
            script_path=base.script_path,
            log_path=base.log_path,
            supervisor_pid_path=base.supervisor_pid_path,
            daemon_pid_path=base.daemon_pid_path,
            supervisor_status_path=base.supervisor_status_path,
            extra_args=tuple(args),
        )


def _profile_option_values(argv: Sequence[str], option: str) -> tuple[str, ...]:
    """Read exact repeated option values from one immutable launch profile."""

    values: list[str] = []
    index = 0
    tokens = tuple(str(item) for item in argv)
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"{option} is missing its launch-profile value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            value = token[len(prefix) :]
            if not value:
                raise ValueError(f"{option} is missing its launch-profile value")
            values.append(value)
        index += 1
    return tuple(values)


class _StableArtifactReadError(RuntimeError):
    """A coordination artifact was unsafe, malformed, or changed while read."""


def _read_stable_regular_bytes(
    path: Path,
    *,
    max_bytes: int = 1_048_576,
) -> tuple[bytes | None, dict[str, Any]]:
    """Read one bounded no-follow regular file with stable inode evidence.

    Callers still hold the artifact's canonical update guard.  The lstat/open
    and post-read identity comparisons additionally reject dangling symlinks,
    hardlinks, and a non-cooperating replace between pathname checks.
    """

    artifact = Path(path)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_nlink),
            int(value.st_uid),
            int(value.st_gid),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    try:
        before = os.lstat(artifact)
    except FileNotFoundError:
        # An open after an absent lstat must also observe absence.  If a
        # non-cooperating writer publishes in that interval, fail closed.
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(artifact, flags)
        except FileNotFoundError:
            try:
                os.lstat(artifact)
            except FileNotFoundError:
                return None, {"state": "absent", "path": str(artifact)}
            raise _StableArtifactReadError(
                f"artifact appeared during absent read: {artifact}"
            ) from None
        except OSError as exc:
            raise _StableArtifactReadError(
                f"cannot prove absent artifact {artifact}: {exc}"
            ) from exc
        else:
            os.close(descriptor)
            raise _StableArtifactReadError(
                f"artifact appeared during absent read: {artifact}"
            )
    except OSError as exc:
        raise _StableArtifactReadError(
            f"cannot lstat coordination artifact {artifact}: {exc}"
        ) from exc

    if stat.S_ISLNK(before.st_mode):
        raise _StableArtifactReadError(
            f"coordination artifact is a symbolic link: {artifact}"
        )
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise _StableArtifactReadError(
            f"coordination artifact is not a single-link regular file: {artifact}"
        )
    if int(before.st_size) < 0 or int(before.st_size) > int(max_bytes):
        raise _StableArtifactReadError(
            f"coordination artifact exceeds its read bound: {artifact}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(artifact, flags)
    except OSError as exc:
        reason = (
            "symbolic link"
            if exc.errno == errno.ELOOP
            else f"open failed: {exc}"
        )
        raise _StableArtifactReadError(
            f"unsafe coordination artifact {artifact}: {reason}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if identity(opened) != identity(before):
            raise _StableArtifactReadError(
                f"coordination artifact changed before open: {artifact}"
            )
        chunks: list[bytes] = []
        remaining = int(max_bytes) + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload_bytes = b"".join(chunks)
        if len(payload_bytes) > int(max_bytes):
            raise _StableArtifactReadError(
                f"coordination artifact exceeds its read bound: {artifact}"
            )
        after_read = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(artifact)
    except OSError as exc:
        raise _StableArtifactReadError(
            f"coordination artifact disappeared during read: {artifact}"
        ) from exc
    if (
        identity(opened) != identity(after_read)
        or identity(opened) != identity(after_path)
        or stat.S_ISLNK(after_path.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or int(after_path.st_nlink) != 1
    ):
        raise _StableArtifactReadError(
            f"coordination artifact changed during read: {artifact}"
        )
    evidence = {
        "state": "present",
        "path": str(artifact),
        "content_sha256": "sha256:" + hashlib.sha256(payload_bytes).hexdigest(),
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "mode": int(opened.st_mode),
        "link_count": int(opened.st_nlink),
        "uid": int(opened.st_uid),
        "gid": int(opened.st_gid),
        "size": int(opened.st_size),
        "mtime_ns": int(opened.st_mtime_ns),
        "ctime_ns": int(opened.st_ctime_ns),
    }
    return payload_bytes, evidence


def _read_stable_regular_json(
    path: Path,
    *,
    max_bytes: int = 1_048_576,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read one exact bounded JSON object without path-following ambiguity."""

    artifact = Path(path)
    payload_bytes, evidence = _read_stable_regular_bytes(
        artifact,
        max_bytes=max_bytes,
    )
    if payload_bytes is None:
        return None, evidence

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StableArtifactReadError(
                    f"coordination artifact has duplicate JSON key {key!r}: "
                    f"{artifact}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _StableArtifactReadError(
            f"coordination artifact is malformed JSON: {artifact}"
        ) from exc
    if not isinstance(payload, dict):
        raise _StableArtifactReadError(
            f"coordination artifact must be a JSON object: {artifact}"
        )
    return dict(payload), evidence


def _managed_daemon_kernel_fence_for_track(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    launch_argv: Sequence[str],
    process_identity: ProcessIdentity,
) -> _ManagedDaemonKernelFence:
    """Build the parent-trusted daemon scope for one plan-bound lane."""

    if (
        process_identity.process_group_id != process_identity.pid
        or process_identity.session_id != process_identity.pid
    ):
        raise ProcessIdentityMismatch(
            "plan-bound lane root does not own a dedicated lifecycle session"
        )

    def last_option(name: str) -> str:
        values = _profile_option_values(launch_argv, name)
        if not values:
            raise ValueError(f"plan-bound daemon fence lacks {name}")
        return values[-1]

    resolved = track.resolve(repo_root)
    raw_state_dir = last_option("--state-dir")
    raw_state_prefix = last_option("--state-prefix")
    raw_todo_path = last_option("--todo-path")
    state_dir = _resolve_path(repo_root, Path(raw_state_dir))
    todo_path = _resolve_path(repo_root, Path(raw_todo_path))
    if (
        state_dir.resolve(strict=False)
        != resolved.daemon_pid_path.parent.resolve(strict=False)
        or not raw_state_prefix
    ):
        raise ValueError("plan-bound daemon marker scope differs from its lane")
    daemon_entrypoint = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon"
    )
    owner_scope = {
        "repo_root": str(repo_root.resolve(strict=False)),
        "state_dir": str(state_dir.resolve(strict=False)),
        "state_prefix": raw_state_prefix,
        "todo_path": str(todo_path.resolve(strict=False)),
        "daemon_entrypoint": daemon_entrypoint,
        "lifecycle_session_id": str(process_identity.session_id),
        "process_group_policy": "dedicated_group_inherited_session",
    }
    return _ManagedDaemonKernelFence(
        pid_path=resolved.daemon_pid_path,
        identity_path=supervised_child_identity_path(
            resolved.daemon_pid_path
        ),
        owner_scope=owner_scope,
        root_session_id=process_identity.session_id,
        state_dir_option=raw_state_dir,
        state_prefix=raw_state_prefix,
        todo_path_option=raw_todo_path,
        daemon_entrypoint=daemon_entrypoint,
    )


def _managed_daemon_kernel_fence_from_profile(
    profile: LifecycleProfile,
    process_identity: ProcessIdentity,
) -> _ManagedDaemonKernelFence:
    """Reconstruct the same trusted projection from immutable launch state."""

    def last_option(name: str) -> str:
        values = _profile_option_values(profile.argv, name)
        if not values:
            raise ValueError(f"plan-bound daemon fence lacks {name}")
        return values[-1]

    state_dir_option = last_option("--state-dir")
    state_prefix = last_option("--state-prefix")
    todo_path_values = _profile_option_values(profile.argv, "--todo-path")
    if len(todo_path_values) > 1:
        raise ValueError("plan-bound daemon fence has duplicate --todo-path")
    todo_path_option = todo_path_values[0] if todo_path_values else None
    state_dir = _resolve_path(
        Path(profile.repository_root),
        Path(state_dir_option),
    )
    todo_path = (
        _resolve_path(Path(profile.repository_root), Path(todo_path_option))
        if todo_path_option is not None
        else None
    )
    if (
        process_identity.process_group_id != process_identity.pid
        or process_identity.session_id != process_identity.pid
        or state_dir.resolve(strict=False)
        != Path(profile.state_root).resolve(strict=False)
        or not state_prefix
    ):
        raise ValueError("plan-bound daemon profile kernel scope is invalid")
    daemon_entrypoint = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon."
        "implementation_daemon"
    )
    pid_path = state_dir / f"{state_prefix}_managed_daemon.pid"
    return _ManagedDaemonKernelFence(
        pid_path=pid_path,
        identity_path=supervised_child_identity_path(pid_path),
        owner_scope=(
            {
                "repo_root": str(
                    Path(profile.repository_root).resolve(strict=False)
                ),
                "state_dir": str(state_dir.resolve(strict=False)),
                "state_prefix": state_prefix,
                "todo_path": str(todo_path.resolve(strict=False)),
                "daemon_entrypoint": daemon_entrypoint,
                "lifecycle_session_id": str(process_identity.session_id),
                "process_group_policy": (
                    "dedicated_group_inherited_session"
                ),
            }
            if todo_path is not None
            else None
        ),
        root_session_id=process_identity.session_id,
        state_dir_option=state_dir_option,
        state_prefix=state_prefix,
        todo_path_option=todo_path_option,
        daemon_entrypoint=daemon_entrypoint,
    )


def _kernel_session_members_once(
    session_id: int,
    *,
    excluded_pids: Sequence[int] = (),
) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    """Observe non-zombie births in one Linux session without procfs environ.

    A session can only be joined by descendants already in that session.  The
    lane root creates it at birth, so membership is a kernel ownership fact
    even when a credential-bearing process is deliberately non-dumpable.
    """

    if session_id <= 1:
        return "unknown", ()
    excluded = {int(item) for item in excluded_pids}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return "unknown", ()
    members: list[tuple[int, int, int]] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            metadata = os.stat(entry, follow_symlinks=False)
            if int(metadata.st_uid) != os.geteuid():
                continue
            raw = (entry / "stat").read_text(encoding="ascii")
            closing = raw.rfind(")")
            fields = raw[closing + 2 :].split()
            state = fields[0]
            process_group = int(fields[2])
            observed_session = int(fields[3])
            start_ticks = int(fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, IndexError, UnicodeError, ValueError):
            return "unknown", ()
        if closing < 0 or start_ticks <= 0:
            return "unknown", ()
        if state == "Z" or observed_session != int(session_id):
            continue
        members.append((pid, start_ticks, process_group))
    return ("alive" if members else "dead"), tuple(sorted(members))


def _managed_daemon_command_matches_kernel_fence(
    command: Sequence[str],
    binding: _ManagedDaemonKernelFence,
) -> bool:
    tokens = tuple(str(item) for item in command)
    if binding.daemon_entrypoint not in tokens:
        return False

    def exact_option(name: str, expected: str) -> bool:
        return _profile_option_values(tokens, name) == (expected,)

    return bool(
        binding.todo_path_option is not None
        and
        exact_option("--state-dir", binding.state_dir_option)
        and exact_option("--state-prefix", binding.state_prefix)
        and exact_option("--todo-path", binding.todo_path_option)
    )


def _managed_daemon_kernel_fence_observation(
    binding: _ManagedDaemonKernelFence,
    *,
    root_identity: ProcessIdentity,
) -> tuple[str, SupervisedChildIdentity | None]:
    """Return ALIVE/DEAD/UNKNOWN for one exact kernel-bound daemon birth."""

    if (
        root_identity.session_id != binding.root_session_id
        or root_identity.process_group_id != root_identity.pid
    ):
        return "unknown", None
    try:
        pid_bytes, _pid_evidence = _read_stable_regular_bytes(
            binding.pid_path,
            max_bytes=32,
        )
        identity_payload, _identity_evidence = _read_stable_regular_json(
            binding.identity_path,
            max_bytes=1_048_576,
        )
    except _StableArtifactReadError:
        return "unknown", None

    session_state, session_members = _kernel_session_members_once(
        binding.root_session_id,
        excluded_pids=(root_identity.pid,),
    )
    if pid_bytes is None and identity_payload is None:
        return (
            ("dead", None)
            if session_state == "dead"
            else ("unknown", None)
        )
    if pid_bytes is None or identity_payload is None:
        return "unknown", None
    if re.fullmatch(rb"[1-9][0-9]*\n", pid_bytes) is None:
        return "unknown", None
    marker_pid = int(pid_bytes[:-1])
    identity = SupervisedChildIdentity.from_dict(identity_payload)
    if (
        identity is None
        or identity.process_birth.pid != marker_pid
        or binding.owner_scope is None
        or dict(identity.owner_scope) != dict(binding.owner_scope)
        or not _managed_daemon_command_matches_kernel_fence(
            identity.command,
            binding,
        )
    ):
        return "unknown", None
    liveness = supervised_child_identity_liveness(identity)
    if liveness is OwnerLiveness.UNKNOWN:
        return "unknown", identity
    if liveness is OwnerLiveness.DEAD:
        return (
            ("dead", identity)
            if session_state == "dead"
            else ("unknown", identity)
        )
    try:
        parent, process_group, session, start_ticks = (
            LinuxProcessAdapter._stat(marker_pid)  # noqa: SLF001
        )
    except (FileNotFoundError, ProcessLookupError):
        return "unknown", identity
    except (OSError, UnicodeError, ValueError):
        return "unknown", identity
    if (
        start_ticks != identity.process_birth.start_time_ticks
        or process_group != marker_pid
        or session != binding.root_session_id
        or not any(
            member_pid == marker_pid and member_start == start_ticks
            for member_pid, member_start, _member_group in session_members
        )
        or read_process_command_argv(marker_pid) != identity.command
    ):
        return "unknown", identity
    # ``parent`` may be the lane root, a subreaper, or init after a crash.  It
    # is deliberately not used as authority; exact inherited session
    # membership survives all three cases.
    del parent
    return "alive", identity


def _fence_managed_daemon_from_kernel_binding(
    binding: _ManagedDaemonKernelFence,
    *,
    root_identity: ProcessIdentity,
    grace_seconds: float,
) -> bool:
    """Fence one exact daemon group; never signal an unproven marker PID."""

    state, identity = _managed_daemon_kernel_fence_observation(
        binding,
        root_identity=root_identity,
    )
    if state == "dead":
        return True
    if state != "alive" or identity is None:
        return False
    # Close a sidecar/PID-reuse race immediately before signalling.  The
    # second observation also repeats exact proc argv and kernel-session proof.
    repeated_state, repeated_identity = (
        _managed_daemon_kernel_fence_observation(
            binding,
            root_identity=root_identity,
        )
    )
    if (
        repeated_state != "alive"
        or repeated_identity is None
        or repeated_identity.record_id != identity.record_id
        or repeated_identity.process_birth != identity.process_birth
    ):
        return False
    fenced = terminate_pid_tree(
        identity.process_birth.pid,
        grace_seconds=max(0.0, float(grace_seconds)),
        freeze_first=True,
        require_gone=True,
        owned_process_group_id=identity.process_birth.pid,
        expected_root_start_time_ticks=(
            identity.process_birth.start_time_ticks
        ),
    )
    if not fenced:
        return False
    for _scan in range(3):
        final_state, _final_identity = (
            _managed_daemon_kernel_fence_observation(
                binding,
                root_identity=root_identity,
            )
        )
        if final_state == "dead":
            return True
        if final_state == "unknown":
            return False
        time.sleep(0.02)
    return False


def _configured_board_gate_relative_path(value: Any, *, field: str) -> str:
    """Return one canonical repository-relative launch declaration."""

    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{field} is not a safe repository-relative path")
    return value


def _configured_board_profile_path(
    root: Path,
    path: Path,
    *,
    stamp: str = "",
    require_regular: bool = False,
) -> str:
    resolved = _resolve_path(root, Path(path))
    _lexical_contained_path(root, resolved, require_regular=require_regular)
    try:
        rendered = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("configured-board launch path escapes repository") from exc
    return rendered.replace(stamp, "{stamp}") if stamp else rendered


_CONFIGURED_BOARD_DATABASE_PROFILE_OPTIONS = (
    "--task-source-kind",
    "--authority-mode",
    "--state-failover-policy",
    "--endpoint-secret-handle",
    "--state-store-id",
    "--state-store-generation",
    "--state-schema-revision",
    "--event-store-path",
    "--runtime-registry-path",
    "--worktree-root",
    "--export-profile",
)


def _configured_board_database_profile(argv: Sequence[str]) -> dict[str, Any]:
    tokens = [str(item) for item in argv]
    if redact_database_program_argv(tokens) != tokens:
        raise ValueError(
            "configured-board launch profile contains raw credential material"
        )
    return {
        "options": {
            option: list(_profile_option_values(tokens, option))
            for option in _CONFIGURED_BOARD_DATABASE_PROFILE_OPTIONS
        },
        "explicit_legacy_task_source_count": tokens.count(
            "--explicit-legacy-task-source"
        ),
    }


def configured_board_live_seal_launch_profile(
    *,
    tracks: Sequence[SupervisorTrack],
    repo_root: Path,
    common_args: Sequence[str],
    python_executable: str,
    stamp: str = "",
) -> dict[str, Any]:
    """Render deterministic dry-run evidence for the disabled live profile."""

    root = _canonical_accepted_tree_root(Path(repo_root))
    requested_python = str(python_executable)
    executable_candidate = requested_python
    if not Path(executable_candidate).is_absolute():
        executable_candidate = shutil.which(executable_candidate) or ""
    if not executable_candidate:
        raise ValueError("configured-board Python executable cannot be resolved")
    executable, executable_sha256 = _python_executable_sha256(
        executable_candidate
    )
    common = tuple(str(item) for item in common_args)
    projected_tracks: list[dict[str, Any]] = []
    names: set[str] = set()
    for track in tracks:
        resolved = track.resolve(root)
        if not resolved.name or resolved.name in names or resolved.module_name:
            raise ValueError(
                "configured-board live profile requires unique exact-script tracks"
            )
        names.add(resolved.name)
        script_bytes, script_evidence = _read_stable_regular_bytes(
            resolved.script_path,
            max_bytes=8_388_608,
        )
        if script_bytes is None:
            raise ValueError("configured-board target script is absent")
        extra_args = tuple(str(item) for item in resolved.extra_args)
        child_argv = [
            requested_python,
            str(resolved.script_path),
            *common,
            *extra_args,
        ]
        projected_tracks.append(
            {
                "name": resolved.name,
                "script_path": _configured_board_profile_path(
                    root, resolved.script_path, require_regular=True
                ),
                "script_sha256": script_evidence["content_sha256"],
                "log_path": _configured_board_profile_path(
                    root, resolved.log_path, stamp=stamp
                ),
                "supervisor_pid_path": _configured_board_profile_path(
                    root, resolved.supervisor_pid_path
                ),
                "daemon_pid_path": _configured_board_profile_path(
                    root, resolved.daemon_pid_path
                ),
                "supervisor_status_path": (
                    _configured_board_profile_path(
                        root, resolved.supervisor_status_path
                    )
                    if resolved.supervisor_status_path is not None
                    else ""
                ),
                "extra_args": list(extra_args),
                "database_profile": _configured_board_database_profile(
                    extra_args
                ),
                "declared_database_program": (
                    resolved.database_program.redacted_dict()
                    if resolved.database_program is not None
                    else None
                ),
                "child_argv_cid": content_identity(
                    {
                        "schema": CONFIGURED_BOARD_LIVE_SEAL_CHILD_SCHEMA,
                        "argv": child_argv,
                    }
                ),
            }
        )
    validator_relative, verifier_relative = next(
        iter(CONFIGURED_BOARD_LIVE_SEAL_VERIFIERS.items())
    )
    source_evidence: dict[str, str] = {}
    for label, relative, bound in (
        ("runner", PLAN_BOUND_GATE_ENTRY_PATH, 8_388_608),
        ("validator", validator_relative, 4_194_304),
        ("live_verifier", verifier_relative, 8_388_608),
    ):
        raw, evidence = _read_stable_regular_bytes(root / relative, max_bytes=bound)
        if raw is None:
            raise ValueError(f"configured-board {label} source is absent")
        source_evidence[f"{label}_path"] = relative
        source_evidence[f"{label}_sha256"] = evidence["content_sha256"]
    combined = (*common, *(arg for track in tracks for arg in track.extra_args))
    return {
        "schema": CONFIGURED_BOARD_LIVE_SEAL_PROFILE_SCHEMA,
        "python_executable": requested_python,
        "python_executable_path": executable,
        "python_executable_sha256": executable_sha256,
        "python_flags_required": ["-I", "-S"],
        "common_args": list(common),
        "database_profile": _configured_board_database_profile(common),
        "source_evidence": source_evidence,
        "tracks": projected_tracks,
        "lane_policy": {
            "track_count": len(projected_tracks),
            "track_names": [item["name"] for item in projected_tracks],
            "strict_task_sharding_count": combined.count(
                "--strict-task-sharding"
            ),
            "task_shards": [
                {
                    "name": item["name"],
                    "count": list(
                        _profile_option_values(source.extra_args, "--task-shard-count")
                    ),
                    "index": list(
                        _profile_option_values(source.extra_args, "--task-shard-index")
                    ),
                }
                # ``strict=`` is unavailable on the supported Python 3.8.
                for item, source in zip(projected_tracks, tracks)  # noqa: B905
            ],
        },
        "launch_policy": {
            "status": "no-go",
            "blocker": CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO,
        },
    }


def _configured_board_live_seal_required(
    common_args: Sequence[str],
    tracks: Sequence[SupervisorTrack] = (),
) -> bool:
    argv = [str(item) for item in common_args]
    argv.extend(str(arg) for track in tracks for arg in track.extra_args)
    return DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION in (
        _profile_option_values(argv, "--state-schema-revision")
    )


def _strict_plan_bound_process_fence_observation(
    profile: LifecycleProfile,
    process_identity: ProcessIdentity,
    *,
    max_scans: int = 3,
) -> tuple[str, Any]:
    """Return ALIVE/DEAD/UNKNOWN without collapsing ``/proc`` failures.

    ``LinuxProcessAdapter`` intentionally offers a convenient boolean API for
    ordinary cleanup.  Same-revision task transfer is an authority boundary:
    an unreadable same-user process or an unstable marker scan must remain
    UNKNOWN and can never be treated as proof of death.
    """

    from ..control.lifecycle_orchestrator import (
        CONFIGURATION_ROOT_ENV,
        PROFILE_ID_ENV,
        REPOSITORY_ROOT_ENV,
        RUN_ID_ENV,
        RUN_ROOT_ENV,
        STATE_ROOT_ENV,
        TARGET_ID_ENV,
        ProcessTreeSnapshot,
    )

    if (
        isinstance(max_scans, bool)
        or not isinstance(max_scans, int)
        or not 2 <= max_scans <= 8
    ):
        raise ValueError("strict process observation scan bound is invalid")
    adapter = LinuxProcessAdapter()
    try:
        _parent, _group, _session, started = adapter._stat(  # noqa: SLF001
            process_identity.pid
        )
    except (FileNotFoundError, ProcessLookupError):
        pass
    except (OSError, UnicodeError, ValueError):
        return "unknown", None
    else:
        if started == process_identity.start_time_ticks:
            return "alive", None

    expected_markers = {
        RUN_ID_ENV: profile.run_id,
        PROFILE_ID_ENV: profile.profile_id,
        TARGET_ID_ENV: profile.target_id,
        REPOSITORY_ROOT_ENV: profile.repository_root,
        STATE_ROOT_ENV: profile.state_root,
        RUN_ROOT_ENV: profile.run_root,
        CONFIGURATION_ROOT_ENV: profile.configuration_root,
    }
    stable_empty_scans = 0
    for _scan in range(max_scans):
        session_state, _session_members = _kernel_session_members_once(
            process_identity.session_id,
            excluded_pids=(process_identity.pid,),
        )
        if session_state == "unknown":
            return "unknown", None
        if session_state == "alive":
            # Unlike the public daemon sidecar, membership in the unique
            # lane session is kernel-enforced.  A reparented, non-dumpable
            # managed daemon therefore remains positive ALIVE evidence.
            return "alive", None
        try:
            entries = tuple(Path("/proc").iterdir())
        except OSError:
            return "unknown", None
        members: list[ProcessIdentity] = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                metadata = os.stat(entry, follow_symlinks=False)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                return "unknown", None
            if int(metadata.st_uid) != os.geteuid():
                continue
            pid = int(entry.name)
            try:
                parent, group, session, _started = adapter._stat(  # noqa: SLF001
                    pid
                )
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, UnicodeError, ValueError):
                return "unknown", None
            try:
                environment = adapter._environ(pid)  # noqa: SLF001
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, UnicodeError, ValueError):
                if (
                    parent == process_identity.pid
                    or group == process_identity.process_group_id
                    or session == process_identity.session_id
                ):
                    return "unknown", None
                continue
            if (
                environment.get(RUN_ID_ENV) != profile.run_id
                or environment.get(TARGET_ID_ENV) != profile.target_id
            ):
                continue
            if any(
                environment.get(name) != value
                for name, value in expected_markers.items()
            ):
                return "unknown", None
            try:
                members.append(adapter._identity(pid, profile))  # noqa: SLF001
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, UnicodeError, ValueError, ProcessIdentityMismatch):
                return "unknown", None
        if members:
            return "alive", ProcessTreeSnapshot(
                profile_id=profile.profile_id,
                run_id=profile.run_id,
                members=tuple(members),
                captured_at_ms=int(time.time() * 1000),
            )
        stable_empty_scans += 1
        if stable_empty_scans >= 2:
            return "dead", ProcessTreeSnapshot(
                profile_id=profile.profile_id,
                run_id=profile.run_id,
                members=(),
                captured_at_ms=int(time.time() * 1000),
            )
    return "unknown", None


def reassign_fenced_plan_bound_child(
    *,
    donor: PlanBoundSupervisorChild,
    recipient: PlanBoundSupervisorChild,
    donor_process: subprocess.Popen[bytes],
    repo_root: Path,
) -> PlanBoundSupervisorChild:
    """CAS-transfer one failed slice using live production fence/claim reads.

    This is the sole production reassignment caller.  The multi-runner owns
    ``donor_process`` and its immutable lifecycle profile.  While the
    canonical ``PlanRevisionStore`` transaction is held, it proves that exact
    process birth and every marker-bound descendant are dead, then holds the
    existing implementation-daemon task-claim update locks until the owner
    pointer is committed.  No caller-authored liveness booleans or second
    selection/claim authority participate.
    """

    from ..control.plan_execution_store import (
        MAX_PLAN_BOUND_WAVE_TRANSFERS,
        ConfiguredBoardExecutionSlices,
        ExecutionClaimConflictError,
        ExecutionSliceViolationError,
        PlanSliceReassignment,
        ProductionParallelPlanAdapter,
        _load_plan_bound_process_birth_chain_locked,
        _secure_store_active,
        _secure_store_cas,
        _secure_store_continuation,
        plan_bound_terminal_missing_key,
        plan_bound_wave_diff_barrier_key,
    )
    from ..merge.checkout_lock import serialized_lock_update
    from ..task_sources.plan_revision_store import PlanRevisionStore
    from ..todo_daemon import implementation_daemon as daemon_module

    if not isinstance(donor, PlanBoundSupervisorChild) or not isinstance(
        recipient, PlanBoundSupervisorChild
    ):
        raise TypeError("slice reassignment requires typed plan-bound children")
    if donor.lane_id == recipient.lane_id:
        raise ExecutionSliceViolationError(
            "slice reassignment requires a distinct recipient lane"
        )
    immutable_fields = (
        "plan_revision_store_path",
        "revision_cid",
        "plan_root_cid",
        "execution_plan_cid",
        "capacity_snapshot_id",
        "slice_manifest_cid",
        "source_head",
        "source_tree",
        "task_source_revision",
        "configuration_root",
        "accepted_tree_root",
    )
    if any(getattr(donor, name) != getattr(recipient, name) for name in immutable_fields):
        raise ExecutionSliceViolationError(
            "reassignment donor and recipient are not in one immutable wave"
        )
    if donor_process.poll() is None:
        raise ExecutionClaimConflictError(
            "cannot reassign a slice whose donor process has not exited"
        )

    profile = getattr(donor_process, "_agent_supervisor_lifecycle_profile", None)
    process_identity = getattr(
        donor_process, "_agent_supervisor_process_identity", None
    )
    launch_process_birth_cid = getattr(
        donor_process,
        "_agent_supervisor_process_birth_cid",
        "",
    )
    if not isinstance(profile, LifecycleProfile) or process_identity is None:
        raise ExecutionClaimConflictError(
            "donor has no production lifecycle birth identity"
        )
    if not isinstance(launch_process_birth_cid, str) or not launch_process_birth_cid:
        raise ExecutionClaimConflictError(
            "donor has no durable gated process-birth evidence"
        )
    # ``ProcessIdentity`` is intentionally imported through the lifecycle
    # module's public object graph: isinstance against a caller mapping is not
    # accepted as process-birth evidence.
    from ..control.lifecycle_orchestrator import ProcessIdentity

    if not isinstance(process_identity, ProcessIdentity):
        raise ExecutionClaimConflictError(
            "donor lifecycle birth evidence has the wrong type"
        )
    try:
        resolved_repo = _canonical_accepted_tree_root(Path(repo_root))
        accepted_tree = _canonical_accepted_tree_root(
            Path(donor.accepted_tree_root)
        )
        profile_repo = _canonical_accepted_tree_root(
            Path(profile.repository_root)
        )
    except ValueError as exc:
        raise ExecutionClaimConflictError(
            "donor repository authority is not a lexical accepted tree"
        ) from exc
    if (
        accepted_tree != resolved_repo
        or profile_repo != resolved_repo
        or process_identity.pid != int(donor_process.pid)
        or process_identity.profile_id != profile.profile_id
        or process_identity.run_id != profile.run_id
        or process_identity.target_id != profile.target_id
        or profile.target_id != f"supervisor-track:{donor.name}"
    ):
        raise ExecutionClaimConflictError(
            "donor process birth is not bound to the failed plan lane"
        )

    exact_profile_options = {
        "--plan-revision-store-path": str(Path(donor.plan_revision_store_path)),
        "--plan-bound-revision-cid": donor.revision_cid,
        "--plan-bound-plan-root-cid": donor.plan_root_cid,
        "--plan-bound-execution-plan-cid": donor.execution_plan_cid,
        "--plan-bound-capacity-snapshot-id": donor.capacity_snapshot_id,
        "--plan-bound-slice-manifest-cid": donor.slice_manifest_cid,
        "--plan-bound-slice-id": donor.slice_id,
        "--plan-bound-source-head": donor.source_head,
        "--plan-bound-source-tree": donor.source_tree,
        "--plan-bound-task-source-revision": donor.task_source_revision,
        "--plan-bound-configuration-root": donor.configuration_root,
        "--plan-bound-accepted-tree-root": str(donor.accepted_tree_root),
        "--plan-bound-lane-id": donor.lane_id,
    }
    if "--plan-bound-dispatch" not in profile.argv:
        raise ExecutionClaimConflictError(
            "donor lifecycle profile is not a plan-bound launch"
        )
    for option, expected in exact_profile_options.items():
        if _profile_option_values(profile.argv, option) != (expected,):
            raise ExecutionClaimConflictError(
                f"donor lifecycle profile changed {option}"
            )
    if (
        _profile_option_values(profile.argv, "--execution-slice-task-id")
        != donor.task_ids
        or _profile_option_values(profile.argv, "--execution-slice-task-cid")
        != donor.task_cids
    ):
        raise ExecutionClaimConflictError(
            "donor process birth carries a different task ID/CID slice"
        )

    store_path = Path(donor.plan_revision_store_path)
    if not store_path.is_absolute():
        store_path = resolved_repo / store_path
    try:
        store_path = _lexical_contained_path(resolved_repo, store_path)
        donor_state_dir = _lexical_contained_path(
            resolved_repo,
            _resolve_path(resolved_repo, Path(donor.state_dir)),
        )
    except ValueError as exc:
        raise ExecutionClaimConflictError(
            "donor state/store authority is not lexical and contained"
        ) from exc
    if (
        donor_state_dir.parent != store_path.parent
        or store_path.name != "plan-revision-store"
    ):
        raise ExecutionClaimConflictError(
            "donor state/store authority crossed the runtime state root"
        )
    store = PlanRevisionStore(store_path)
    plan_adapter = ProductionParallelPlanAdapter(plan_revision_store=store)
    # The real daemon method is the canonical filename relation.  A read-only
    # probe avoids constructing unrelated worktree/provider runtime state.
    claim_probe = object.__new__(daemon_module.PortalImplementationDaemon)
    claim_probe.repo_root = resolved_repo

    with store._thread_lock:  # noqa: SLF001 - canonical store transaction
        with store._guard():  # noqa: SLF001 - canonical cross-process guard
            active = _secure_store_active(store)
            terminal_barrier = _secure_store_continuation(
                store,
                plan_bound_wave_diff_barrier_key(
                    donor.revision_cid,
                    donor.slice_manifest_cid,
                ),
            )
            if terminal_barrier is not None:
                raise ExecutionClaimConflictError(
                    "cannot reassign after the wave barrier terminalized"
                )
            if _secure_store_continuation(
                store,
                plan_bound_terminal_missing_key(
                    donor.revision_cid,
                    donor.slice_id,
                ),
            ) is not None:
                raise ExecutionClaimConflictError(
                    "cannot reassign a terminal-missing slice"
                )
            if active is None or active.revision_cid != donor.revision_cid:
                raise ExecutionClaimConflictError(
                    "slice reassignment requires the exact active revision"
                )
            if active.plan_root_cid != donor.plan_root_cid:
                raise ExecutionClaimConflictError(
                    "slice reassignment observed a mixed active plan root"
                )
            revision_payload = _secure_store_cas(store, donor.revision_cid)
            from ..planning.plan_revision_contracts import PlanRevision

            revision = PlanRevision.from_dict(revision_payload)
            if revision.to_dict() != revision_payload:
                raise ExecutionClaimConflictError(
                    "active revision changed during typed decode"
                )
            if revision.materialization_transaction_cid != donor.slice_manifest_cid:
                raise ExecutionClaimConflictError(
                    "active revision does not own the slice manifest"
                )
            manifest = ConfiguredBoardExecutionSlices.from_dict(
                _secure_store_cas(store, donor.slice_manifest_cid)
            )
            matches = tuple(
                item for item in manifest.slices if item.slice_id == donor.slice_id
            )
            if len(matches) != 1:
                raise ExecutionSliceViolationError(
                    "slice reassignment target is absent or duplicated"
                )
            execution_slice = matches[0]
            if (
                execution_slice.task_ids != donor.task_ids
                or execution_slice.task_cids != donor.task_cids
            ):
                raise ExecutionSliceViolationError(
                    "donor population differs from the immutable slice"
                )
            launch_birth_binding = _load_plan_bound_process_birth_chain_locked(
                store,
                revision_cid=donor.revision_cid,
                slice_id=donor.slice_id,
                lane_id=donor.lane_id,
            )
            if (
                launch_birth_binding is None
                or launch_birth_binding[0] != launch_process_birth_cid
            ):
                raise ExecutionClaimConflictError(
                    "donor durable process birth is not the current chain head"
                )
            launch_birth = launch_birth_binding[1]
            if (
                launch_birth.revision_cid != donor.revision_cid
                or launch_birth.slice_manifest_cid != donor.slice_manifest_cid
                or launch_birth.slice_id != donor.slice_id
                or launch_birth.lane_id != donor.lane_id
                or launch_birth.task_ids != donor.task_ids
                or launch_birth.task_cids != donor.task_cids
                or launch_birth.profile != profile.to_dict()
                or launch_birth.process_birth != process_identity.to_dict()
            ):
                raise ExecutionClaimConflictError(
                    "donor durable process birth is mixed"
                )
            current = plan_adapter._load_slice_reassignment_locked(  # noqa: SLF001
                revision_cid=donor.revision_cid,
                slice_id=donor.slice_id,
            )
            current_cid = current[0] if current is not None else ""
            current_owner = (
                current[1].recipient_lane_id
                if current is not None
                else execution_slice.lane_id
            )
            generation = current[1].generation + 1 if current is not None else 1
            if current_cid != donor.reassignment_cid:
                raise ExecutionClaimConflictError(
                    "slice reassignment CAS lost to another lane"
                )
            if current_owner != donor.lane_id:
                raise ExecutionSliceViolationError(
                    "declared donor no longer owns the slice"
                )
            wave_transfer_budget = min(
                MAX_PLAN_BOUND_WAVE_TRANSFERS,
                max(1, len(manifest.nonempty)),
            )
            wave_transfer_count = 0
            for manifest_slice in manifest.nonempty:
                observed_reassignment = (
                    plan_adapter._load_slice_reassignment_locked(  # noqa: SLF001
                        revision_cid=donor.revision_cid,
                        slice_id=manifest_slice.slice_id,
                    )
                )
                if observed_reassignment is not None:
                    wave_transfer_count += observed_reassignment[1].generation
            if wave_transfer_count + 1 > wave_transfer_budget:
                raise ExecutionClaimConflictError(
                    "wave reassignment budget is exhausted"
                )
            visited_lanes = {execution_slice.lane_id}
            cursor = current[1] if current is not None else None
            cursor_cids: set[str] = set()
            while cursor is not None:
                visited_lanes.update(
                    {cursor.donor_lane_id, cursor.recipient_lane_id}
                )
                prior_cid = cursor.prior_reassignment_cid
                if not prior_cid:
                    break
                if prior_cid in cursor_cids:
                    raise ExecutionClaimConflictError(
                        "slice reassignment chain cycles during recipient check"
                    )
                cursor_cids.add(prior_cid)
                cursor = PlanSliceReassignment.from_dict(
                    _secure_store_cas(store, prior_cid)
                )
            if recipient.lane_id in visited_lanes:
                raise ExecutionClaimConflictError(
                    "slice reassignment recipient already owned this slice"
                )

            # Re-observe the exact captured birth while the CAS guard is held.
            # An empty marker-selected tree proves the root and all inherited
            # children were fenced, not merely that a numeric PID disappeared.
            process_state, fenced_tree = (
                _strict_plan_bound_process_fence_observation(
                    profile,
                    process_identity,
                )
            )
            if process_state == "alive":
                raise ExecutionClaimConflictError(
                    "donor process birth remains alive"
                )
            if process_state != "dead" or fenced_tree is None:
                raise ExecutionClaimConflictError(
                    "donor process death is not provable"
                )
            if fenced_tree.members:
                raise ExecutionClaimConflictError(
                    "donor marker-bound process tree is not fenced"
                )
            process_evidence = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "plan-slice-donor-fence@1"
                ),
                "revision_cid": donor.revision_cid,
                "slice_manifest_cid": donor.slice_manifest_cid,
                "slice_id": donor.slice_id,
                "donor_lane_id": donor.lane_id,
                "donor_track_name": donor.name,
                "profile": profile.to_dict(),
                "process_birth": process_identity.to_dict(),
                "fenced_tree": fenced_tree.to_dict(),
                "launch_process_birth_cid": launch_process_birth_cid,
            }
            donor_process_birth_cid = store.put_cas(process_evidence)
            if _secure_store_cas(store, donor_process_birth_cid) != process_evidence:
                raise ExecutionClaimConflictError(
                    "donor process-birth evidence failed CAS round trip"
                )

            # Work may move only before the donor has consumed any attempt.
            # The daemon durably charges this canonical state projection in
            # the same write that marks an implementation active, before
            # provider dispatch.  Missing state is pristine; malformed,
            # active, prior-attempt, or completed state all fail closed.
            donor_state_path = (
                donor_state_dir / f"{donor.state_prefix}_task_state.json"
            )
            try:
                attempt_state, attempt_state_identity = (
                    _read_stable_regular_json(donor_state_path)
                )
            except _StableArtifactReadError as exc:
                raise ExecutionClaimConflictError(
                    "cannot prove canonical donor attempt state pristine"
                ) from exc
            attempt_payload = dict(attempt_state or {})
            raw_attempts = attempt_payload.get("implementation_attempts", {})
            raw_cid_attempts = attempt_payload.get(
                "implementation_attempts_by_cid", {}
            )
            if not isinstance(raw_attempts, Mapping) or not isinstance(
                raw_cid_attempts, Mapping
            ):
                raise ExecutionClaimConflictError(
                    "canonical donor attempt counters are malformed"
                )

            def attempt_counts(value: Mapping[Any, Any]) -> dict[str, int]:
                result: dict[str, int] = {}
                for raw_key, raw_count in value.items():
                    key = str(raw_key).strip()
                    if (
                        not key
                        or isinstance(raw_count, bool)
                        or not isinstance(raw_count, int)
                        or raw_count < 0
                    ):
                        raise ExecutionClaimConflictError(
                            "canonical donor attempt counter is malformed"
                        )
                    result[key] = int(raw_count)
                return result

            display_attempts = attempt_counts(raw_attempts)
            cid_attempts = attempt_counts(raw_cid_attempts)
            raw_active_attempt = attempt_payload.get("active_attempt", 0)
            if (
                isinstance(raw_active_attempt, bool)
                or not isinstance(raw_active_attempt, int)
                or raw_active_attempt < 0
            ):
                raise ExecutionClaimConflictError(
                    "canonical donor active attempt is malformed"
                )
            active_attempt = raw_active_attempt
            raw_implementation_in_progress = attempt_payload.get(
                "implementation_in_progress", False
            )
            if not isinstance(raw_implementation_in_progress, bool):
                raise ExecutionClaimConflictError(
                    "canonical donor implementation-in-progress flag is malformed"
                )
            completed_ids = attempt_payload.get("completed_task_ids", []) or []
            if isinstance(completed_ids, (str, bytes)) or not isinstance(
                completed_ids, Sequence
            ):
                raise ExecutionClaimConflictError(
                    "canonical donor completion projection is malformed"
                )
            prior_effect_markers = (
                active_attempt > 0
                or raw_implementation_in_progress
                or bool(str(attempt_payload.get("active_task_id") or "").strip())
                or bool(str(attempt_payload.get("active_task_cid") or "").strip())
                or bool(
                    str(
                        attempt_payload.get("last_implementation_task_id") or ""
                    ).strip()
                )
                or bool(
                    str(
                        attempt_payload.get("last_implementation_task_cid") or ""
                    ).strip()
                )
                or bool(
                    str(
                        attempt_payload.get("last_implementation_started_at")
                        or ""
                    ).strip()
                )
                or bool(
                    str(
                        attempt_payload.get("last_implementation_finished_at")
                        or ""
                    ).strip()
                )
                or attempt_payload.get("last_implementation_returncode")
                is not None
                or bool(
                    str(
                        attempt_payload.get("last_implementation_log_path")
                        or ""
                    ).strip()
                )
                or bool(
                    str(
                        attempt_payload.get("last_implementation_worktree_path")
                        or ""
                    ).strip()
                )
                or bool(
                    str(
                        attempt_payload.get("last_implementation_branch") or ""
                    ).strip()
                )
                or bool(
                    str(
                        attempt_payload.get("last_implementation_commit") or ""
                    ).strip()
                )
                or bool(attempt_payload.get("last_proof_workflow"))
                or bool(
                    str(attempt_payload.get("last_merge_started_at") or "").strip()
                )
                or bool(
                    str(attempt_payload.get("last_merge_finished_at") or "").strip()
                )
                or bool(
                    str(attempt_payload.get("last_merge_branch") or "").strip()
                )
                or bool(
                    str(attempt_payload.get("last_merge_commit") or "").strip()
                )
                or attempt_payload.get("last_merge_returncode") is not None
                or any(count > 0 for count in display_attempts.values())
                or any(count > 0 for count in cid_attempts.values())
                or bool(set(map(str, completed_ids)).intersection(donor.task_ids))
            )
            if prior_effect_markers:
                raise ExecutionClaimConflictError(
                    "donor slice has a consumed or active implementation attempt"
                )
            attempt_evidence = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "plan-slice-attempt-absence@1"
                ),
                "revision_cid": donor.revision_cid,
                "slice_manifest_cid": donor.slice_manifest_cid,
                "slice_id": donor.slice_id,
                "task_ids": list(execution_slice.task_ids),
                "task_cids": list(execution_slice.task_cids),
                "state_path": str(donor_state_path),
                "state_identity": attempt_state_identity,
                "state": attempt_payload,
                "never_attempted": True,
            }
            attempt_absence_cid = store.put_cas(attempt_evidence)
            if _secure_store_cas(store, attempt_absence_cid) != attempt_evidence:
                raise ExecutionClaimConflictError(
                    "donor attempt-absence evidence failed CAS round trip"
                )

            claim_rows: list[dict[str, Any]] = []
            claim_paths: list[tuple[Path, str, str]] = []
            for task_id, task_cid in zip(
                execution_slice.task_ids,
                execution_slice.task_cids,
                strict=True,
            ):
                claim_path = daemon_module.PortalImplementationDaemon._implementation_task_claim_path(  # noqa: SLF001
                    claim_probe,
                    task_id,
                    canonical_task_cid=task_cid,
                )
                claim_paths.append((claim_path, task_id, task_cid))
            if len({path for path, _task_id, _task_cid in claim_paths}) != len(
                claim_paths
            ):
                raise ExecutionClaimConflictError(
                    "canonical task claim paths are not one-to-one"
                )

            # Claim update guards remain held through continuation publication.
            # Therefore no daemon can acquire one of these exact task CIDs in
            # the gap between the absence observation and owner transfer.
            with ExitStack() as claim_guards:
                for claim_path, _task_id, _task_cid in sorted(
                    claim_paths, key=lambda item: str(item[0])
                ):
                    claim_guards.enter_context(
                        serialized_lock_update(claim_path)
                    )
                for claim_path, task_id, task_cid in claim_paths:
                    try:
                        metadata, artifact_identity = (
                            _read_stable_regular_json(claim_path)
                        )
                    except _StableArtifactReadError as exc:
                        raise ExecutionClaimConflictError(
                            "canonical task claim artifact is unsafe"
                        ) from exc
                    # A stale record still proves that this task crossed the
                    # claim boundary.  Same-revision transfer is deliberately
                    # restricted to truly never-claimed work so it cannot
                    # replay a provider effect or consumed attempt.
                    if metadata is not None:
                        raise ExecutionClaimConflictError(
                            "canonical task claim was already published"
                        )
                    claim_rows.append(
                        {
                            "task_id": task_id,
                            "task_cid": task_cid,
                            "claim_path": str(claim_path),
                            "state": "absent",
                            "artifact_identity": artifact_identity,
                        }
                    )
                claim_evidence = {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "plan-slice-claim-absence@1"
                    ),
                    "revision_cid": donor.revision_cid,
                    "slice_manifest_cid": donor.slice_manifest_cid,
                    "slice_id": donor.slice_id,
                    "task_ids": list(execution_slice.task_ids),
                    "task_cids": list(execution_slice.task_cids),
                    "claims": claim_rows,
                }
                claim_absence_cid = store.put_cas(claim_evidence)
                if _secure_store_cas(store, claim_absence_cid) != claim_evidence:
                    raise ExecutionClaimConflictError(
                        "task-claim absence evidence failed CAS round trip"
                    )
                # Close the observation/publication interval.  Canonical
                # writers remain excluded by their update guards; this second
                # stable read also rejects a non-cooperating path swap that
                # became lstat-visible before owner publication.
                for claim_path, task_id, task_cid in claim_paths:
                    try:
                        final_metadata, final_identity = (
                            _read_stable_regular_json(claim_path)
                        )
                    except _StableArtifactReadError as exc:
                        raise ExecutionClaimConflictError(
                            "canonical task claim changed before reassignment"
                        ) from exc
                    expected_row = next(
                        row
                        for row in claim_rows
                        if row["task_id"] == task_id
                        and row["task_cid"] == task_cid
                    )
                    if (
                        final_metadata is not None
                        or final_identity != expected_row["artifact_identity"]
                    ):
                        raise ExecutionClaimConflictError(
                            "canonical task claim changed before reassignment"
                        )
                try:
                    final_attempt_state, final_attempt_identity = (
                        _read_stable_regular_json(donor_state_path)
                    )
                except _StableArtifactReadError as exc:
                    raise ExecutionClaimConflictError(
                        "canonical donor attempt state changed before reassignment"
                    ) from exc
                if (
                    dict(final_attempt_state or {}) != attempt_payload
                    or final_attempt_identity != attempt_state_identity
                ):
                    raise ExecutionClaimConflictError(
                        "canonical donor attempt state changed before reassignment"
                    )
                reassignment = PlanSliceReassignment(
                    revision_cid=donor.revision_cid,
                    plan_root_cid=active.plan_root_cid,
                    slice_manifest_cid=donor.slice_manifest_cid,
                    slice_id=donor.slice_id,
                    donor_lane_id=donor.lane_id,
                    recipient_lane_id=recipient.lane_id,
                    task_ids=execution_slice.task_ids,
                    task_cids=execution_slice.task_cids,
                    generation=generation,
                    prior_reassignment_cid=current_cid,
                    donor_process_birth_cid=donor_process_birth_cid,
                    attempt_absence_cid=attempt_absence_cid,
                    claim_absence_cid=claim_absence_cid,
                )
                reassignment_cid = store.put_cas(reassignment.to_dict())
                key = plan_adapter._reassignment_key(  # noqa: SLF001
                    donor.revision_cid, donor.slice_id
                )
                store.put_continuation(
                    key,
                    {
                        "phase": "committed",
                        "operation": "slice_reassignment",
                        "revision_cid": donor.revision_cid,
                        "plan_root_cid": active.plan_root_cid,
                        "slice_id": donor.slice_id,
                        "reassignment_cid": reassignment_cid,
                        "generation": generation,
                    },
                )
                observed = plan_adapter._load_slice_reassignment_locked(  # noqa: SLF001
                    revision_cid=donor.revision_cid,
                    slice_id=donor.slice_id,
                )
                if observed is None or observed != (
                    reassignment_cid,
                    reassignment,
                ):
                    raise ExecutionClaimConflictError(
                        "slice reassignment CAS did not publish exactly"
                    )

    suffix = hashlib.sha256(
        f"{donor.slice_id}:{generation}:{recipient.lane_id}".encode()
    ).hexdigest()[:12]
    return replace(
        donor,
        name=f"{recipient.name}-steal-{generation}-{suffix}",
        state_dir=recipient.state_dir,
        state_prefix=f"{recipient.state_prefix}-steal-{generation}-{suffix}",
        lane_id=recipient.lane_id,
        reassignment_cid=reassignment_cid,
    )


@dataclass(frozen=True)
class ImplementationSupervisorNamespaceTrackSpec:
    """Minimal namespace-based inputs for one implementation-supervisor track."""

    name: str
    script_path: Path | str
    namespace: str
    state_prefix: str | None = None


def implementation_supervisor_namespace_track_config(
    *,
    name: str,
    script_path: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    state_prefix: str | None = None,
) -> ImplementationSupervisorTrackConfig:
    """Return a track config using the standard namespace state directory."""

    return ImplementationSupervisorTrackConfig(
        name=name,
        script_path=script_path,
        state_dir=namespace_paths.state_dir,
        state_prefix=state_prefix or namespace_paths.namespace,
    )


def _implementation_supervisor_namespace_track_spec(
    spec: (
        ImplementationSupervisorNamespaceTrackSpec
        | tuple[str, Path | str, str]
        | tuple[str, Path | str, str, str]
    ),
) -> ImplementationSupervisorNamespaceTrackSpec:
    if isinstance(spec, ImplementationSupervisorNamespaceTrackSpec):
        return spec
    if len(spec) == 3:
        name, script_path, namespace = spec
        state_prefix = None
    elif len(spec) == 4:
        name, script_path, namespace, state_prefix = spec
    else:
        raise ValueError(
            "namespace track specs must have NAME|SCRIPT|NAMESPACE or "
            "NAME|SCRIPT|NAMESPACE|STATE_PREFIX"
        )
    return ImplementationSupervisorNamespaceTrackSpec(
        name=name,
        script_path=script_path,
        namespace=namespace,
        state_prefix=state_prefix,
    )


def implementation_supervisor_namespace_track_configs(
    *,
    repo_root: Path | str,
    track_specs: Sequence[
        ImplementationSupervisorNamespaceTrackSpec
        | tuple[str, Path | str, str]
        | tuple[str, Path | str, str, str]
    ],
    data_root: Path | str = "data",
) -> tuple[ImplementationSupervisorTrackConfig, ...]:
    """Return implementation-supervisor track configs from namespace-based specs."""

    from ..core.wrapper_utils import agent_supervisor_namespace_paths

    return tuple(
        implementation_supervisor_namespace_track_config(
            name=resolved_spec.name,
            script_path=resolved_spec.script_path,
            namespace_paths=agent_supervisor_namespace_paths(
                repo_root,
                resolved_spec.namespace,
                data_root=data_root,
            ),
            state_prefix=resolved_spec.state_prefix,
        )
        for resolved_spec in (
            _implementation_supervisor_namespace_track_spec(spec) for spec in track_specs
        )
    )


@dataclass(frozen=True)
class ConfiguredMultiSupervisorCliRunner:
    """Project-bound CLI argv for launching the reusable multi-supervisor runner."""

    argv: tuple[str, ...]

    def args(self) -> list[str]:
        """Return the configured runner argv as a mutable list."""

        return list(self.argv)

    def run(self, extra_argv: Sequence[str] | None = None) -> int:
        """Run the multi-supervisor CLI with configured args plus any overrides."""

        return main([*self.argv, *(extra_argv or ())])

    def run_cli(self, argv: Sequence[str] | None = None) -> int:
        """Run from a wrapper CLI, defaulting overrides from ``sys.argv``."""

        return self.run(sys.argv[1:] if argv is None else argv)


@dataclass(frozen=True)
class ConfiguredMultiSupervisorLauncher:
    """Prepared launcher for a configured multi-supervisor runner."""

    runner: ConfiguredMultiSupervisorCliRunner
    env_defaults: tuple[tuple[str, str], ...] = ()
    prepare_environment: Callable[[], None] | None = None

    def args(self) -> list[str]:
        """Return the configured runner argv as a mutable list."""

        return self.runner.args()

    def prepare(self) -> None:
        """Apply environment defaults and run the optional preparation callback."""

        if self.env_defaults:
            apply_env_defaults(dict(self.env_defaults))
        if self.prepare_environment is not None:
            self.prepare_environment()

    def run(self, extra_argv: Sequence[str] | None = None) -> int:
        """Prepare the environment and run the configured multi-supervisor CLI."""

        self.prepare()
        return self.runner.run(extra_argv)

    def run_cli(self, argv: Sequence[str] | None = None) -> int:
        """Prepare and run from a wrapper CLI, defaulting overrides from ``sys.argv``."""

        self.prepare()
        return self.runner.run_cli(argv)


class SupervisorRunInterrupted(Exception):
    """Raised internally when a signal requests orderly shutdown."""


class PlanBoundProcessBirthError(RuntimeError):
    """A plan-bound child was fenced before launch authority was released."""

    def __init__(
        self,
        message: str,
        *,
        pid: int,
        profile: LifecycleProfile,
        all_trees_fenced: bool,
    ) -> None:
        super().__init__(message)
        self.pid = int(pid)
        self.profile = profile
        self.profile_id = profile.profile_id
        self.all_trees_fenced = bool(all_trees_fenced)


def utc_run_stamp() -> str:
    """Return a UTC run stamp suitable for log/pid filenames."""

    return datetime.now(_UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_timestamp() -> str:
    """Return a compact local timestamp for operator logs."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_track_spec(spec: str, *, stamp: str = "") -> SupervisorTrack:
    """Parse ``NAME|SCRIPT|LOG|SUPERVISOR_PID|DAEMON_PID[|SUPERVISOR_STATUS]`` specs."""

    rendered = spec.format(stamp=stamp) if stamp else spec
    parts = rendered.split("|")
    if len(parts) not in {5, 6} or not parts[0].strip():
        raise ValueError(
            "track specs must have NAME|SCRIPT|LOG|SUPERVISOR_PID|DAEMON_PID"
            "[|SUPERVISOR_STATUS]"
        )
    name, script, log, supervisor_pid, daemon_pid = (part.strip() for part in parts[:5])
    supervisor_status = parts[5].strip() if len(parts) == 6 else ""
    return SupervisorTrack(
        name=name,
        script_path=Path(script),
        log_path=Path(log),
        supervisor_pid_path=Path(supervisor_pid),
        daemon_pid_path=Path(daemon_pid),
        supervisor_status_path=Path(supervisor_status) if supervisor_status else None,
    )


def implementation_supervisor_track_spec(
    *,
    name: str,
    script_path: Path | str,
    state_dir: Path | str,
    state_prefix: str,
) -> str:
    """Return a standard implementation-supervisor track spec."""

    state_path = Path(state_dir).as_posix()
    return "|".join(
        (
            str(name),
            Path(script_path).as_posix(),
            f"{state_path}/{state_prefix}_8h_run_{{stamp}}.log",
            f"{state_path}/{state_prefix}_supervisor.pid",
            f"{state_path}/{state_prefix}_managed_daemon.pid",
        )
    )


def implementation_supervisor_compact_track_spec(
    *,
    name: str,
    script_path: Path | str,
    state_dir: Path | str,
    state_prefix: str,
) -> str:
    """Return a compact ``NAME|SCRIPT|STATE_DIR|STATE_PREFIX`` implementation-track spec."""

    return "|".join(
        (
            str(name),
            Path(script_path).as_posix(),
            Path(state_dir).as_posix(),
            str(state_prefix),
        )
    )


def implementation_supervisor_compact_track_specs(
    track_configs: Sequence[ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]],
) -> tuple[str, ...]:
    """Return compact implementation-track specs from structured track configs."""

    specs: list[str] = []
    for config in track_configs:
        if isinstance(config, ImplementationSupervisorTrackConfig):
            specs.append(config.compact_spec())
            continue
        name, script_path, state_dir, state_prefix = config
        specs.append(
            implementation_supervisor_compact_track_spec(
                name=name,
                script_path=script_path,
                state_dir=state_dir,
                state_prefix=state_prefix,
            )
        )
    return tuple(specs)


def parse_implementation_track_spec(
    spec: str,
    *,
    stamp: str = "",
    database_program: DatabaseProgramConfig | None = None,
) -> SupervisorTrack:
    """Parse ``NAME|SCRIPT|STATE_DIR|STATE_PREFIX`` implementation-track specs."""

    parts = [part.strip() for part in spec.split("|")]
    if len(parts) != 4 or not parts[0]:
        raise ValueError("implementation track specs must have NAME|SCRIPT|STATE_DIR|STATE_PREFIX")
    name, script, state_dir, state_prefix = parts
    track = parse_track_spec(
        implementation_supervisor_track_spec(
            name=name,
            script_path=script,
            state_dir=state_dir,
            state_prefix=state_prefix,
        ),
        stamp=stamp,
    )
    extra_args: list[str] = [
        "--state-dir",
        str(state_dir),
        "--state-prefix",
        str(state_prefix),
    ]
    if database_program is not None:
        extra_args.extend(database_program.cli_args())
    return SupervisorTrack(
        name=track.name,
        script_path=track.script_path,
        log_path=track.log_path,
        supervisor_pid_path=track.supervisor_pid_path,
        daemon_pid_path=track.daemon_pid_path,
        supervisor_status_path=Path(state_dir) / f"{state_prefix}_supervisor_status.json",
        extra_args=tuple(extra_args),
        database_program=database_program,
    )


def expand_implementation_track_lanes(
    spec: str,
    *,
    stamp: str = "",
    lanes_per_track: int = 1,
    database_program: DatabaseProgramConfig | None = None,
) -> list[SupervisorTrack]:
    """Return one or more deterministic shard lanes for an implementation-track spec."""

    lanes = max(1, int(lanes_per_track))
    if lanes == 1:
        return [
            parse_implementation_track_spec(
                spec,
                stamp=stamp,
                database_program=database_program,
            )
        ]

    parts = [part.strip() for part in spec.split("|")]
    if len(parts) != 4 or not parts[0]:
        raise ValueError("implementation track specs must have NAME|SCRIPT|STATE_DIR|STATE_PREFIX")
    name, script, state_dir, state_prefix = parts
    tracks: list[SupervisorTrack] = []
    for index in range(lanes):
        lane_state_dir = Path(state_dir) / f"lane-{index}"
        lane_state_prefix = f"{state_prefix}_lane_{index}"
        track = parse_implementation_track_spec(
            implementation_supervisor_compact_track_spec(
                name=f"{name}-{index}",
                script_path=script,
                state_dir=lane_state_dir,
                state_prefix=lane_state_prefix,
            ),
            stamp=stamp,
            database_program=database_program,
        )
        tracks.append(
            SupervisorTrack(
                name=track.name,
                script_path=track.script_path,
                log_path=track.log_path,
                supervisor_pid_path=track.supervisor_pid_path,
                daemon_pid_path=track.daemon_pid_path,
                supervisor_status_path=track.supervisor_status_path,
                extra_args=(
                    *track.extra_args,
                    "--task-shard-count",
                    str(lanes),
                    "--task-shard-index",
                    str(index),
                ),
                database_program=database_program,
            )
        )
    return tracks


def expand_database_implementation_track_lanes(
    track: DatabaseImplementationTrack,
    *,
    stamp: str = "",
    lanes_per_track: int = 1,
) -> list[SupervisorTrack]:
    """Expand a database-bound implementation track into isolated lanes."""

    return expand_implementation_track_lanes(
        track.track_config().compact_spec(),
        stamp=stamp,
        lanes_per_track=max(1, int(lanes_per_track)),
        database_program=track.database_program,
    )


def supervisor_track_payload(track: SupervisorTrack) -> dict[str, str]:
    """Return a serializable track description for tests and diagnostics."""

    payload = {
        "name": track.name,
        "script_path": str(track.script_path),
        "log_path": str(track.log_path),
        "supervisor_pid_path": str(track.supervisor_pid_path),
        "daemon_pid_path": str(track.daemon_pid_path),
    }
    if track.module_name:
        payload["module_name"] = track.module_name
    return payload


def dynamic_bundle_scheduler_track(
    *,
    name: str,
    bundle_index_path: Path | str,
    state_root: Path | str,
    max_lanes: int,
    repo_root: Path | str = Path("."),
    poll_interval: float = 5.0,
    implement: bool = True,
    claimant_did: str = "did:web:ipfs-accelerate.local",
) -> SupervisorTrack:
    """Build a managed track for the persistent leased bundle scheduler.

    Unlike deterministic ``lanes_per_track`` shards, this is one scheduler
    process that continuously lends a bounded number of slots to live work.
    """

    if int(max_lanes) < 1:
        raise ValueError("max_lanes must be at least 1")
    root = Path(state_root)
    return SupervisorTrack(
        name=str(name),
        script_path=Path("."),
        log_path=root / "bundle_scheduler.log",
        supervisor_pid_path=root / "bundle_scheduler.pid",
        daemon_pid_path=root / "bundle_scheduler_worker.pid",
        module_name="ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor",
        extra_args=(
            "--repo-root", str(repo_root),
            "--bundle-index-path", str(bundle_index_path),
            "--state-root", str(state_root),
            "--max-lanes", str(max_lanes),
            "--poll-interval", str(poll_interval),
            "--claimant-did", str(claimant_did),
            "--start",
            "--implement" if implement else "--no-implement",
        ),
    )


def _env_default_items(
    defaults: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(defaults, Mapping):
        iterable = defaults.items()
    else:
        iterable = defaults
    return tuple((str(name), str(value)) for name, value in iterable)


def _env_default_value(value: bool | int | str) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def seal_ordered_implementation_provider_route(
    environment: MutableMapping[str, str] | None = None,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, str]:
    """Classify a direct Codex selector or seal the reviewed ordered route.

    Validation precedes every mutation.  A lone ``codex`` provider remains
    the legacy direct-provider selector consumed by the implementation
    daemon.  An unset route receives all six ordered bindings together.
    Compatible legacy Grok primary aliases are canonicalized to ``grok_cli``;
    any incompatible ordered tuple fails closed and leaves the environment
    unchanged.
    """

    target = os.environ if environment is None else environment
    route_environment = {
        name: str(target.get(name, "") or "").strip()
        for name in ORDERED_IMPLEMENTATION_PROVIDER_ROUTE
    }
    authorization_environment = {
        name: str(target.get(name, "") or "").strip()
        for name in _ROUTE_AUTHORIZATION_ENV_NAMES
    }
    selected_name = route_environment[_IMPLEMENTATION_PROVIDER_ENV].lower()
    if (
        selected_name in {"codex", "auto"}
        and not any(
            value
            for name, value in route_environment.items()
            if name != _IMPLEMENTATION_PROVIDER_ENV
        )
        and not any(authorization_environment.values())
    ):
        selected_provider = {_IMPLEMENTATION_PROVIDER_ENV: selected_name}
        target.update(selected_provider)
        return selected_provider
    authorization = None
    if any(authorization_environment.values()):
        if not all(authorization_environment.values()):
            raise ValueError(
                "scoped agent route authorization environment is incomplete"
            )
        authorization = load_agent_implementation_route_authorization(
            repo_root=(Path.cwd() if repo_root is None else repo_root),
            artifact_path=authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_PATH"
            ],
            board_namespace=authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_BOARD_NAMESPACE"
            ],
            expected_sha256=authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_SHA256"
            ],
            expected_authorization_id=authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_ID"
            ],
        )
        if (
            authorization.authorization_kind
            != authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_AUTHORIZATION_KIND"
            ]
            or authorization.source_head
            != authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_HEAD"
            ]
            or authorization.source_tree
            != authorization_environment[
                "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_SOURCE_TREE"
            ]
        ):
            raise ValueError("scoped agent route authorization binding drifted")
    plan = resolve_agent_implementation_route(
        primary_provider_id=route_environment[
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
        ],
        primary_model_id=route_environment[
            "IPFS_ACCELERATE_AGENT_GROK_MODEL"
        ],
        fallback_provider_id=route_environment[
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
        ],
        fallback_model_id=route_environment[
            "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
        ],
        fallback_trigger=route_environment[
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
        ],
        fallback_reasoning_effort=route_environment[
            "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
        ],
        default_route="legacy",
        authorization=authorization,
    )
    if authorization is not None and plan.route_id != authorization_environment[
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_ID"
    ]:
        raise ValueError("scoped agent implementation route identity drifted")
    selected_route = plan.as_environment()
    target.update(selected_route)
    return selected_route


def implementation_multi_supervisor_env_defaults(
    *,
    python_unbuffered: bool | int | str | None = True,
    grok_merge_resolver_timeout_seconds: int | str | None = 900,
    codex_merge_resolver_timeout_seconds: int | str | None = 600,
    # Retained as ignored keyword-only compatibility shims. Copilot is not a
    # member of the reviewed merge-resolver route.
    copilot_merge_resolver_timeout_seconds: int | str | None = None,
    prefer_copilot_merge_resolver: bool | int | str | None = None,
) -> dict[str, str]:
    """Return reusable environment defaults for long-running implementation supervisors."""

    del copilot_merge_resolver_timeout_seconds, prefer_copilot_merge_resolver
    defaults: dict[str, str] = {}
    if python_unbuffered is not None:
        defaults["PYTHONUNBUFFERED"] = _env_default_value(python_unbuffered)
    if grok_merge_resolver_timeout_seconds is not None:
        defaults["GROK_MERGE_RESOLVER_TIMEOUT_SECONDS"] = _env_default_value(
            grok_merge_resolver_timeout_seconds
        )
    if codex_merge_resolver_timeout_seconds is not None:
        defaults["CODEX_MERGE_RESOLVER_TIMEOUT_SECONDS"] = _env_default_value(
            codex_merge_resolver_timeout_seconds
        )
    # Emit the complete ordered route as one atomic default.  Supplying only
    # the Codex model would make legacy ``auto`` selection treat Codex as the
    # primary and bypass the Grok quota gate.
    defaults.update(ORDERED_IMPLEMENTATION_PROVIDER_ROUTE)
    return defaults


def build_configured_multi_supervisor_cli_runner(
    *,
    repo_root: Path | str,
    duration_seconds: float | int | str = 28800.0,
    duration_seconds_env_var: str = "",
    heartbeat_interval_seconds: float | int | str | None = None,
    supervisor_status_stale_seconds: float | int | str | None = None,
    stop_grace_seconds: float | int | str | None = None,
    stamp: str = "",
    stamp_env_var: str = "",
    master_dir: Path | str = Path("data/agent_supervisor"),
    master_log: Path | str | None = None,
    master_pid_path: Path | str | None = None,
    label: str = "multi-supervisor",
    python_executable: str = "python3",
    implementation_supervisor_defaults: bool = False,
    implementation_supervisor_command: str = "",
    implementation_supervisor_llm_merge_resolver_command: str = "",
    implementation_tracks: Sequence[str] = (),
    implementation_track_configs: Sequence[
        ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]
    ] = (),
    plan_bound_tracks: Sequence[PlanBoundSupervisorChild] = (),
    tracks: Sequence[str] = (),
    common_args: Sequence[str] = (),
    detach: bool = False,
    database_program: DatabaseProgramConfig | None = None,
) -> ConfiguredMultiSupervisorCliRunner:
    """Build reusable multi-supervisor CLI argv from project-specific tracks."""

    _ = database_program  # optional pin; track configs may carry per-track program


    effective_duration_seconds = (
        env_str(duration_seconds_env_var, str(duration_seconds))
        if duration_seconds_env_var
        else duration_seconds
    )
    effective_stamp_default = stamp or utc_run_stamp()
    effective_stamp = (
        env_str(stamp_env_var, effective_stamp_default)
        if stamp_env_var
        else effective_stamp_default
    )
    argv = [
        "--repo-root",
        str(repo_root),
        "--duration-seconds",
        str(effective_duration_seconds),
        "--stamp",
        effective_stamp,
        "--master-dir",
        str(master_dir),
        "--label",
        label,
        "--python-executable",
        python_executable,
    ]
    if heartbeat_interval_seconds is not None:
        argv.extend(["--heartbeat-interval-seconds", str(heartbeat_interval_seconds)])
    if supervisor_status_stale_seconds is not None:
        argv.extend(["--supervisor-status-stale-seconds", str(supervisor_status_stale_seconds)])
    if stop_grace_seconds is not None:
        argv.extend(["--stop-grace-seconds", str(stop_grace_seconds)])
    if master_log is not None:
        argv.extend(["--master-log", str(master_log)])
    if master_pid_path is not None:
        argv.extend(["--master-pid-path", str(master_pid_path)])
    if implementation_supervisor_defaults:
        argv.append("--implementation-supervisor-defaults")
    if implementation_supervisor_command:
        argv.extend(["--implementation-supervisor-command", implementation_supervisor_command])
    if implementation_supervisor_llm_merge_resolver_command:
        argv.extend(
            [
                "--implementation-supervisor-llm-merge-resolver-command",
                implementation_supervisor_llm_merge_resolver_command,
            ]
        )
    for track in tracks:
        argv.extend(["--track", str(track)])
    for track in implementation_tracks:
        argv.extend(["--implementation-track", str(track)])
    for track in implementation_supervisor_compact_track_specs(implementation_track_configs):
        argv.extend(["--implementation-track", str(track)])
    for track in plan_bound_tracks:
        if not isinstance(track, PlanBoundSupervisorChild):
            raise TypeError("plan_bound_tracks must contain PlanBoundSupervisorChild")
        argv.extend(["--implementation-plan-bound-track", track.cli_record()])
    if plan_bound_tracks:
        argv.append("--plan-bound-wave")
    for arg in common_args:
        argv.append(f"--common-arg={arg}")
    if detach:
        argv.append("--detach")
    return ConfiguredMultiSupervisorCliRunner(tuple(argv))


def build_configured_multi_supervisor_launcher(
    *,
    repo_root: Path | str,
    duration_seconds: float | int | str = 28800.0,
    duration_seconds_env_var: str = "",
    heartbeat_interval_seconds: float | int | str | None = None,
    supervisor_status_stale_seconds: float | int | str | None = None,
    stop_grace_seconds: float | int | str | None = None,
    stamp: str = "",
    stamp_env_var: str = "",
    master_dir: Path | str = Path("data/agent_supervisor"),
    master_log: Path | str | None = None,
    master_pid_path: Path | str | None = None,
    label: str = "multi-supervisor",
    python_executable: str = "python3",
    implementation_supervisor_defaults: bool = False,
    implementation_supervisor_command: str = "",
    implementation_supervisor_llm_merge_resolver_command: str = "",
    implementation_tracks: Sequence[str] = (),
    implementation_track_configs: Sequence[
        ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]
    ] = (),
    plan_bound_tracks: Sequence[PlanBoundSupervisorChild] = (),
    tracks: Sequence[str] = (),
    common_args: Sequence[str] = (),
    detach: bool = False,
    env_defaults: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredMultiSupervisorLauncher:
    """Build a prepared multi-supervisor launcher from project-specific inputs."""

    return ConfiguredMultiSupervisorLauncher(
        runner=build_configured_multi_supervisor_cli_runner(
            repo_root=repo_root,
            duration_seconds=duration_seconds,
            duration_seconds_env_var=duration_seconds_env_var,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            supervisor_status_stale_seconds=supervisor_status_stale_seconds,
            stop_grace_seconds=stop_grace_seconds,
            stamp=stamp,
            stamp_env_var=stamp_env_var,
            master_dir=master_dir,
            master_log=master_log,
            master_pid_path=master_pid_path,
            label=label,
            python_executable=python_executable,
            implementation_supervisor_defaults=implementation_supervisor_defaults,
            implementation_supervisor_command=implementation_supervisor_command,
            implementation_supervisor_llm_merge_resolver_command=(
                implementation_supervisor_llm_merge_resolver_command
            ),
            implementation_tracks=implementation_tracks,
            implementation_track_configs=implementation_track_configs,
            plan_bound_tracks=plan_bound_tracks,
            tracks=tracks,
            common_args=common_args,
            detach=detach,
        ),
        env_defaults=_env_default_items(env_defaults),
        prepare_environment=prepare_environment,
    )


def build_repo_implementation_multi_supervisor_launcher(
    *,
    repo_root: Path | str,
    implementation_track_configs: Sequence[
        ImplementationSupervisorTrackConfig | tuple[str, Path | str, Path | str, str]
    ],
    resolver_script_path: Path | str = "",
    implementation_supervisor_command: str = "",
    implementation_supervisor_llm_merge_resolver_command: str = "",
    duration_seconds: float | int | str = 28800.0,
    duration_seconds_env_var: str = "DURATION_SECONDS",
    heartbeat_interval_seconds: float | int | str | None = None,
    supervisor_status_stale_seconds: float | int | str | None = None,
    stop_grace_seconds: float | int | str | None = None,
    stamp: str = "",
    stamp_env_var: str = "STAMP",
    master_dir: Path | str = Path("data/agent_supervisor"),
    master_log: Path | str | None = None,
    master_pid_path: Path | str | None = None,
    label: str = "implementation supervisor run",
    python_executable: str = "python3",
    common_args: Sequence[str] = (),
    detach: bool = False,
    env_defaults: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    prepare_environment: Callable[[], None] | None = None,
    runtime_package_names: Sequence[Path | str] | None = ("ipfs_accelerate", "ipfs_datasets"),
    runtime_external_dir: Path | str = "external",
    runtime_env_var: str = "PYTHONPATH",
) -> ConfiguredMultiSupervisorLauncher:
    """Build a repo-local implementation multi-supervisor launcher."""

    from ..core.wrapper_utils import (
        build_repo_runtime_environment_callbacks,
        repo_script_command,
    )
    from ..integrations.llm_merge_resolver_fallback import (
        llm_merge_resolver_fallback_command,
    )

    llm_merge_resolver_command = implementation_supervisor_llm_merge_resolver_command
    if not llm_merge_resolver_command and resolver_script_path:
        llm_merge_resolver_command = repo_script_command(repo_root, resolver_script_path)
    if not llm_merge_resolver_command:
        llm_merge_resolver_command = llm_merge_resolver_fallback_command(
            python_executable=python_executable
        )
    effective_prepare_environment = prepare_environment
    if effective_prepare_environment is None and runtime_package_names is not None:
        runtime_environment = build_repo_runtime_environment_callbacks(
            repo_root,
            package_names=runtime_package_names,
            external_dir=runtime_external_dir,
            env_var=runtime_env_var,
        )
        effective_prepare_environment = runtime_environment.ensure_pythonpath
    provided_env_defaults = dict(_env_default_items(env_defaults))
    route_environment_names = (
        *ORDERED_IMPLEMENTATION_PROVIDER_ROUTE,
        *_ROUTE_AUTHORIZATION_ENV_NAMES,
    )
    caller_route_defaults = {
        name: provided_env_defaults[name]
        for name in route_environment_names
        if name in provided_env_defaults
    }
    sealed_route_defaults = seal_ordered_implementation_provider_route(
        caller_route_defaults,
        repo_root=repo_root,
    )
    if sealed_route_defaults.keys() == {_IMPLEMENTATION_PROVIDER_ENV} and (
        sealed_route_defaults.get(_IMPLEMENTATION_PROVIDER_ENV) in {"codex", "auto"}
    ):
        # Direct Codex and automatic host-CLI selectors must not be overlaid
        # onto the ordered Grok-to-Codex defaults.  Doing so creates a hybrid
        # six-field tuple that the canonical route resolver correctly rejects.
        effective_env_defaults = implementation_multi_supervisor_env_defaults()
        for name in ORDERED_IMPLEMENTATION_PROVIDER_ROUTE:
            effective_env_defaults.pop(name, None)
    else:
        effective_env_defaults = implementation_multi_supervisor_env_defaults()
    effective_env_defaults.update(
        {
            name: value
            for name, value in provided_env_defaults.items()
            if name not in route_environment_names
        }
    )
    effective_env_defaults.update(sealed_route_defaults)
    return build_configured_multi_supervisor_launcher(
        repo_root=repo_root,
        duration_seconds=duration_seconds,
        duration_seconds_env_var=duration_seconds_env_var,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        supervisor_status_stale_seconds=supervisor_status_stale_seconds,
        stop_grace_seconds=stop_grace_seconds,
        stamp=stamp,
        stamp_env_var=stamp_env_var,
        master_dir=master_dir,
        master_log=master_log,
        master_pid_path=master_pid_path,
        label=label,
        python_executable=python_executable,
        implementation_supervisor_defaults=True,
        implementation_supervisor_command=implementation_supervisor_command,
        implementation_supervisor_llm_merge_resolver_command=llm_merge_resolver_command,
        implementation_track_configs=implementation_track_configs,
        common_args=common_args,
        detach=detach,
        env_defaults=effective_env_defaults,
        prepare_environment=effective_prepare_environment,
    )


def implementation_supervisor_common_args(
    *,
    implementation_command: str = "",
    llm_merge_resolver_command: str = "",
    stale_seconds: int = 1800,
    check_interval: int = 60,
    daemon_interval: int = 120,
    implementation_timeout: int = 1800,
    implementation_log_stall_seconds: int = 900,
    max_restarts: int = 0,
    objective_scan_min_open_tasks: int = 20,
    objective_scan_max_findings: int = 12,
    objective_scan_cooldown_seconds: int = 900,
    objective_refill_timeout_seconds: int = 600,
    objective_surplus_findings_per_goal: int = 6,
    objective_surplus_min_terms_per_todo: int = 4,
    codebase_scan_cooldown_seconds: int = 900,
    codebase_refill_timeout_seconds: int = 600,
    llm_merge_resolver_timeout_seconds: int = 1800,
    strict_task_sharding: bool = False,
    idle_lane_work_stealing: str = "",
) -> list[str]:
    """Return standard common args for long-running implementation supervisors."""

    args = [
        "--implement",
        "--objective-refill-scan",
        "--codebase-refill-scan",
        "--stale-seconds",
        str(stale_seconds),
        "--check-interval",
        str(check_interval),
        "--daemon-interval",
        str(daemon_interval),
        "--implementation-timeout",
        str(implementation_timeout),
        "--implementation-log-stall-seconds",
        str(implementation_log_stall_seconds),
        "--max-restarts",
        str(max_restarts),
        "--objective-scan-min-open-tasks",
        str(objective_scan_min_open_tasks),
        "--objective-scan-max-findings",
        str(objective_scan_max_findings),
        "--objective-scan-cooldown-seconds",
        str(objective_scan_cooldown_seconds),
        "--objective-refill-timeout-seconds",
        str(objective_refill_timeout_seconds),
        "--objective-surplus-findings-per-goal",
        str(objective_surplus_findings_per_goal),
        "--objective-surplus-min-terms-per-todo",
        str(objective_surplus_min_terms_per_todo),
        "--codebase-scan-cooldown-seconds",
        str(codebase_scan_cooldown_seconds),
        "--codebase-refill-timeout-seconds",
        str(codebase_refill_timeout_seconds),
        "--llm-merge-resolver-timeout-seconds",
        str(llm_merge_resolver_timeout_seconds),
    ]
    if implementation_command:
        args.extend(["--implementation-command", implementation_command])
    if llm_merge_resolver_command:
        args.extend(["--llm-merge-resolver-command", llm_merge_resolver_command])
    if strict_task_sharding:
        args.append("--strict-task-sharding")
    if idle_lane_work_stealing:
        args.extend(
            ["--idle-lane-work-stealing", idle_lane_work_stealing]
        )
    return args


def _emit(output: OutputFn, message: str) -> None:
    output(f"{iso_timestamp()} {message}")


def _default_output(message: str) -> None:
    print(message, flush=True)


def _remove_stale_pid_marker_if_unchanged(pid_path: Path, stale_pid: int) -> bool:
    """Remove a dead PID marker only if it still names the dead process."""

    current_pid = read_pid_file(pid_path)
    if current_pid != stale_pid or pid_alive(current_pid):
        return False
    return remove_runtime_marker(pid_path)


def _remove_owned_pid_projection(pid_path: Path, expected_pid: int) -> bool:
    """Remove a PID projection only while it still names this runner.

    Unlike :func:`_remove_stale_pid_marker_if_unchanged`, this helper may be
    used by the still-running master process during its orderly teardown.  A
    changed marker is never removed, so a concurrently started replacement
    retains its projection.
    """

    try:
        with serialized_lock_update(pid_path):
            payload, evidence = _read_stable_regular_bytes(
                pid_path,
                max_bytes=32,
            )
            if payload != f"{int(expected_pid)}\n".encode("ascii"):
                return False
            observed = os.lstat(pid_path)
            if (
                evidence.get("state") != "present"
                or int(evidence.get("device", -1)) != int(observed.st_dev)
                or int(evidence.get("inode", -1)) != int(observed.st_ino)
                or stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or int(observed.st_nlink) != 1
                or int(observed.st_uid) != os.geteuid()
                or stat.S_IMODE(observed.st_mode) & 0o022
            ):
                return False
            pid_path.unlink()
            return True
    except (_StableArtifactReadError, OSError, UnicodeError, ValueError):
        return False


def _fsync_pid_projection_parent(path: Path) -> None:
    """Durably publish one PID-projection namespace transition."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISDIR(observed.st_mode):
            raise ValueError("PID projection parent is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_private_pid_audit(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one immutable owner-only PID recovery artifact."""

    target = Path(path)
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    temporary_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        temporary_identity = (int(opened.st_dev), int(opened.st_ino))
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("PID recovery audit write stalled")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        # A hard-link publication is atomic and refuses an existing target.
        # The temporary link is removed before the receipt is admitted below.
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        temporary_identity = None
        _fsync_pid_projection_parent(target)
        observed_payload, evidence = _read_stable_regular_bytes(
            target,
            max_bytes=max(4096, len(encoded)),
        )
        if (
            observed_payload != encoded
            or int(evidence.get("uid", -1)) != os.geteuid()
            or int(evidence.get("link_count", -1)) != 1
            or stat.S_IMODE(int(evidence.get("mode", 0))) != 0o600
        ):
            raise ValueError("PID recovery audit publication is not owner-only")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_identity is not None:
            try:
                observed = os.lstat(temporary)
            except FileNotFoundError:
                pass
            else:
                if (
                    (int(observed.st_dev), int(observed.st_ino))
                    == temporary_identity
                    and stat.S_ISREG(observed.st_mode)
                    and int(observed.st_nlink) == 1
                    and int(observed.st_uid) == os.geteuid()
                ):
                    temporary.unlink()


def _pid_projection_audit_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the stable bounded fields admitted into a recovery receipt."""

    return {
        field: evidence[field]
        for field in (
            "path",
            "content_sha256",
            "device",
            "inode",
            "mode",
            "link_count",
            "uid",
            "gid",
            "size",
            "mtime_ns",
            "ctime_ns",
        )
    }


def _quarantine_stale_detached_master_pid_locked(pid_path: Path) -> dict[str, Any]:
    """Quarantine one exact legacy PID after an ESRCH-only absence proof.

    The caller holds ``serialized_lock_update(pid_path)``.  Signal zero probes
    process existence without delivering a signal; every result except ESRCH
    is deliberately treated as live or unknown and therefore non-reclaimable.
    """

    path = Path(pid_path)
    try:
        payload, evidence = _read_stable_regular_bytes(
            path,
            max_bytes=_LEGACY_MASTER_PID_MAX_BYTES,
        )
    except _StableArtifactReadError as exc:
        raise ValueError(f"unsafe detached master PID projection: {exc}") from exc
    if payload is None:
        raise ValueError("detached master PID projection disappeared during recovery")
    if (
        int(evidence.get("uid", -1)) != os.geteuid()
        or int(evidence.get("link_count", -1)) != 1
        or not stat.S_ISREG(int(evidence.get("mode", 0)))
    ):
        raise ValueError(
            "detached master PID projection is not an owned single-link regular file"
        )
    if _LEGACY_MASTER_PID_PAYLOAD.fullmatch(payload) is None:
        raise ValueError("detached master PID projection is not a strict legacy PID")
    legacy_pid = int(payload[:-1].decode("ascii"))
    try:
        os.kill(legacy_pid, 0)
    except ProcessLookupError as exc:
        if exc.errno != errno.ESRCH:
            raise ValueError(
                "detached master PID liveness is unknown"
            ) from exc
    except PermissionError as exc:
        raise ValueError("detached master PID liveness is unknown") from exc
    except OSError as exc:
        raise ValueError("detached master PID liveness is unknown") from exc
    else:
        raise ValueError("detached master PID projection names a live process")

    # Bind the absence proof to the still-identical projection before any
    # pathname mutation.  A non-cooperating replacement fails closed.
    try:
        confirmed_payload, confirmed_evidence = _read_stable_regular_bytes(
            path,
            max_bytes=_LEGACY_MASTER_PID_MAX_BYTES,
        )
    except _StableArtifactReadError as exc:
        raise ValueError(
            f"detached master PID projection changed after liveness proof: {exc}"
        ) from exc
    if confirmed_payload != payload or confirmed_evidence != evidence:
        raise ValueError("detached master PID projection changed after liveness proof")

    observed_at_unix_ns = time.time_ns()
    quarantine_key = hashlib.sha256(
        json.dumps(
            {
                "legacy_pid": legacy_pid,
                "projection": _pid_projection_audit_evidence(evidence),
                "observed_at_unix_ns": observed_at_unix_ns,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    quarantine_path = path.with_name(
        f".{path.name}.stale-{quarantine_key}.quarantine"
    )
    decision_path = path.with_name(
        f".{path.name}.stale-{quarantine_key}.decision.json"
    )
    receipt_path = path.with_name(
        f".{path.name}.stale-{quarantine_key}.receipt.json"
    )
    for target in (quarantine_path, decision_path, receipt_path):
        try:
            os.lstat(target)
        except FileNotFoundError:
            continue
        raise ValueError("detached master PID quarantine target already exists")

    liveness_evidence = {
        "operation": "os.kill",
        "signal": 0,
        "result": "dead",
        "errno": "ESRCH",
        "errno_number": errno.ESRCH,
    }
    decision = {
        "schema": STALE_DETACHED_MASTER_PID_DECISION_SCHEMA,
        "producer": "multi-supervisor-runner@1",
        "model_created": False,
        "completion_authority": False,
        "decision": "quarantine_authorized",
        "legacy_pid": legacy_pid,
        "source_projection": _pid_projection_audit_evidence(evidence),
        "quarantine_path": str(quarantine_path),
        "liveness_evidence": liveness_evidence,
        "observed_at_unix_ns": observed_at_unix_ns,
    }
    decision["decision_receipt_id"] = content_identity(decision)
    # Publish the decision first: a crash can leave an authorization without
    # an outcome claim, but can never leave an unaudited quarantine.
    _publish_private_pid_audit(decision_path, decision)

    latest_payload, latest_evidence = _read_stable_regular_bytes(
        path,
        max_bytes=_LEGACY_MASTER_PID_MAX_BYTES,
    )
    if latest_payload != payload or latest_evidence != evidence:
        raise ValueError("detached master PID projection changed before quarantine")
    try:
        os.rename(path, quarantine_path)
    except OSError as exc:
        raise ValueError("cannot atomically quarantine stale master PID") from exc
    _fsync_pid_projection_parent(path)

    quarantined_payload, quarantined_evidence = _read_stable_regular_bytes(
        quarantine_path,
        max_bytes=_LEGACY_MASTER_PID_MAX_BYTES,
    )
    stable_fields = (
        "content_sha256",
        "device",
        "inode",
        "mode",
        "link_count",
        "uid",
        "gid",
        "size",
        "mtime_ns",
    )
    if (
        quarantined_payload != payload
        or any(
            quarantined_evidence.get(field) != evidence.get(field)
            for field in stable_fields
        )
    ):
        raise ValueError("quarantined master PID projection identity changed")
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("stale master PID pathname remained after quarantine")

    receipt = {
        "schema": STALE_DETACHED_MASTER_PID_RECEIPT_SCHEMA,
        "producer": "multi-supervisor-runner@1",
        "model_created": False,
        "completion_authority": False,
        "outcome": "quarantined",
        "legacy_pid": legacy_pid,
        "decision_receipt_id": decision["decision_receipt_id"],
        "source_projection": _pid_projection_audit_evidence(evidence),
        "quarantine_projection": _pid_projection_audit_evidence(
            quarantined_evidence
        ),
        "liveness_evidence": liveness_evidence,
        "observed_at_unix_ns": observed_at_unix_ns,
    }
    receipt["receipt_id"] = content_identity(receipt)
    _publish_private_pid_audit(receipt_path, receipt)
    return receipt


def _reserve_owned_pid_projection_locked(
    pid_path: Path,
) -> tuple[int, tuple[int, int]]:
    """Reserve one projection while its canonical update lock is held."""

    path = Path(pid_path)
    _require_absent_pid_projection(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ValueError("cannot reserve plan-bound PID projection") from exc
    os.fchmod(descriptor, 0o600)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or int(opened.st_nlink) != 1
        or int(opened.st_uid) != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise ValueError("plan-bound PID reservation is not owner-only")
    _fsync_pid_projection_parent(path)
    return descriptor, (int(opened.st_dev), int(opened.st_ino))


def _reserve_owned_pid_projection(
    pid_path: Path,
) -> tuple[int, tuple[int, int]]:
    """Reserve a no-follow, owner-only PID projection before process birth."""

    path = Path(pid_path)
    with serialized_lock_update(path):
        return _reserve_owned_pid_projection_locked(path)


def _require_absent_pid_projection(pid_path: Path) -> None:
    """Reject every existing PID projection before authority-bearing work."""

    try:
        existing = os.lstat(pid_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("cannot inspect plan-bound PID projection") from exc
    if stat.S_ISLNK(existing.st_mode):
        kind = "symbolic link"
    elif not stat.S_ISREG(existing.st_mode):
        kind = "non-regular file"
    elif int(existing.st_nlink) != 1:
        kind = "hardlinked file"
    else:
        kind = "existing file"
    raise ValueError(f"plan-bound PID projection is an unsafe {kind}")


def _publish_reserved_pid_projection(
    pid_path: Path,
    descriptor: int,
    identity: tuple[int, int],
    pid: int,
) -> None:
    """Publish a PID only while fd and pathname retain the reservation."""

    payload = f"{int(pid)}\n".encode("ascii")
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise ValueError("plan-bound PID projection write was incomplete")
        written += count
    os.fsync(descriptor)
    opened = os.fstat(descriptor)
    observed = os.lstat(pid_path)
    if (
        (int(opened.st_dev), int(opened.st_ino)) != identity
        or (int(observed.st_dev), int(observed.st_ino)) != identity
        or stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or int(observed.st_nlink) != 1
        or int(observed.st_uid) != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or int(observed.st_size) != len(payload)
    ):
        raise ValueError("plan-bound PID projection changed during publication")


def _discard_reserved_pid_projection(
    pid_path: Path,
    identity: tuple[int, int],
) -> None:
    """Remove only the pathname that still owns a failed reservation."""

    with serialized_lock_update(pid_path):
        _discard_reserved_pid_projection_locked(pid_path, identity)


def _discard_reserved_pid_projection_locked(
    pid_path: Path,
    identity: tuple[int, int],
) -> None:
    """Discard the still-identical reservation while its lock is held."""

    try:
        observed = os.lstat(pid_path)
    except FileNotFoundError:
        return
    if (
        (int(observed.st_dev), int(observed.st_ino)) == identity
        and stat.S_ISREG(observed.st_mode)
        and int(observed.st_nlink) == 1
        and int(observed.st_uid) == os.geteuid()
        and stat.S_IMODE(observed.st_mode) == 0o600
    ):
        pid_path.unlink()
        _fsync_pid_projection_parent(pid_path)


def _adopt_or_create_current_master_pid_projection(pid_path: Path) -> None:
    """Adopt this runner or recover a proven-dead foreground predecessor."""

    path = Path(pid_path)
    expected = f"{os.getpid()}\n".encode("ascii")
    with serialized_lock_update(path):
        try:
            payload, evidence = _read_stable_regular_bytes(path, max_bytes=32)
        except _StableArtifactReadError as exc:
            raise ValueError(f"unsafe master PID projection: {exc}") from exc
        if payload is not None:
            if (
                payload == expected
                and int(evidence.get("uid", -1)) == os.geteuid()
                and int(evidence.get("link_count", -1)) == 1
                and stat.S_ISREG(int(evidence.get("mode", 0)))
                and stat.S_IMODE(int(evidence.get("mode", 0))) == 0o600
            ):
                return
            # Foreground runners use the same audited recovery contract as a
            # detached launch.  It admits only a same-UID, single-link regular
            # legacy PID whose signal-zero probe returns ESRCH, then publishes
            # the decision and outcome receipts around an atomic quarantine.
            # Live, permission-denied, malformed, swapped, or ambiguous
            # projections continue to fail closed before any child starts.
            _quarantine_stale_detached_master_pid_locked(path)
        descriptor, identity = _reserve_owned_pid_projection_locked(path)
        try:
            _publish_reserved_pid_projection(
                path,
                descriptor,
                identity,
                os.getpid(),
            )
        except BaseException:
            _discard_reserved_pid_projection_locked(path, identity)
            raise
        finally:
            os.close(descriptor)


def daemon_pid_health_fields(
    pid_path: Path,
    *,
    cleanup_stale_marker: bool = False,
) -> dict[str, object]:
    """Return heartbeat fields for a managed daemon PID marker."""

    daemon_pid = read_pid_file(pid_path)
    if not daemon_pid:
        return {"daemon_pid": None, "daemon_status": "missing"}
    if pid_alive(daemon_pid):
        return {"daemon_pid": daemon_pid, "daemon_status": "live"}
    removed = False
    if cleanup_stale_marker:
        removed = _remove_stale_pid_marker_if_unchanged(pid_path, daemon_pid)
    return {
        "daemon_pid": None,
        "daemon_status": "stale",
        "stale_daemon_pid": daemon_pid,
        "removed_stale_daemon_pid_file": removed,
    }


def format_daemon_heartbeat_fields(fields: Mapping[str, object]) -> str:
    """Return compact daemon health fields for master heartbeat logs."""

    daemon_pid = fields.get("daemon_pid")
    parts = [f"daemon_pid={daemon_pid if daemon_pid else 'unknown'}"]
    stale_pid = fields.get("stale_daemon_pid")
    if stale_pid:
        parts.append(f"stale_daemon_pid={stale_pid}")
    status = fields.get("daemon_status")
    if status and status != "live":
        parts.append(f"daemon_status={status}")
    if fields.get("removed_stale_daemon_pid_file"):
        parts.append("removed_stale_daemon_pid_file=true")
    return " ".join(parts)


def _read_json_dict(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_status_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    return parsed.astimezone(_UTC)


def _inferred_supervisor_status_path(track: SupervisorTrack) -> Path | None:
    if track.supervisor_status_path is not None:
        return track.supervisor_status_path
    name = track.supervisor_pid_path.name
    suffix = "_supervisor.pid"
    if name.endswith(suffix):
        prefix = name[: -len(suffix)]
        return track.supervisor_pid_path.with_name(f"{prefix}_supervisor_status.json")
    return None


def _track_supervisor_status_startup_grace_seconds(
    track: SupervisorTrack,
    *,
    common_args: Sequence[str],
    fallback_seconds: float,
) -> float:
    """Resolve the current launch generation's declared startup grace."""

    launch_args = (
        tuple(track.extra_args)
        if track.module_name
        else (*common_args, *track.extra_args)
    )
    configured = _profile_option_values(
        launch_args,
        "--watchdog-startup-grace-seconds",
    )
    if len(configured) > 1:
        raise ValueError(
            "supervisor status startup grace must be declared at most once"
        )
    raw = configured[0] if configured else fallback_seconds
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "supervisor status startup grace must be a finite non-negative number"
        ) from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "supervisor status startup grace must be a finite non-negative number"
        )
    return seconds


def _relative_or_absolute_path(repo_root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else repo_root / path


def _track_task_state_path(track: SupervisorTrack, *, repo_root: Path) -> Path | None:
    """Resolve a track's task-state projection without trusting an escape path."""

    resolved = track.resolve(repo_root)
    state_root = resolved.supervisor_pid_path.parent.resolve(strict=False)
    status = _read_json_dict(_inferred_supervisor_status_path(resolved))
    candidate = _relative_or_absolute_path(
        repo_root,
        status.get("current_status_path")
        or status.get("progress_path")
        or status.get("state_path"),
    )
    if candidate is None:
        name = resolved.supervisor_pid_path.name
        suffix = "_supervisor.pid"
        if not name.endswith(suffix):
            return None
        candidate = resolved.supervisor_pid_path.with_name(
            f"{name[:-len(suffix)]}_task_state.json"
        )
    candidate = candidate.resolve(strict=False)
    return candidate if _path_within(candidate, state_root) else None


def terminal_task_state_fields(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    fresh_after_epoch_seconds: float,
) -> dict[str, object]:
    """Return fail-closed terminal-quiescence fields for one implementation track.

    Freshness is mandatory.  This prevents a prior completed projection from
    terminating a new run before its child has observed a changed board.
    Launchers should preflight already-completed boards instead of starting a
    timed runner solely to rediscover old terminal state.
    """

    path = _track_task_state_path(track, repo_root=repo_root)
    if path is None:
        return {"terminal_quiescent": False, "task_state_status": "untracked"}
    payload = _read_json_dict(path)
    if not payload:
        return {
            "terminal_quiescent": False,
            "task_state_status": "missing",
            "task_state_path": str(path),
        }
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        modified_at = 0.0
    fresh = modified_at + 1e-6 >= float(fresh_after_epoch_seconds)
    task_count = int(payload.get("task_count") or 0)
    completed_count = int(payload.get("completed_count") or 0)
    active_task_id = str(payload.get("active_task_id") or "").strip()
    implementation_in_progress = bool(payload.get("implementation_in_progress"))
    eligible_ready_count = int(payload.get("eligible_ready_count") or 0)
    blocked_count = int(payload.get("blocked_count") or 0)
    external_reserved_count = int(payload.get("external_reserved_count") or 0)
    slice_task_ids = tuple(
        track.extra_args[index + 1]
        for index, item in enumerate(track.extra_args[:-1])
        if item == "--execution-slice-task-id"
    )
    if slice_task_ids:
        statuses_payload = payload.get("task_statuses")
        statuses = (
            {
                str(key): str(value).strip().lower()
                for key, value in statuses_payload.items()
            }
            if isinstance(statuses_payload, Mapping)
            else {}
        )
        terminal_statuses = {
            "blocked",
            "cancelled",
            "complete",
            "completed",
            "done",
            "failed",
            "quarantined",
            "skipped",
        }
        terminal = bool(
            fresh
            and len(set(slice_task_ids)) == len(slice_task_ids)
            and all(
                statuses.get(task_id) in terminal_statuses
                for task_id in slice_task_ids
            )
            and not active_task_id
            and not implementation_in_progress
        )
    else:
        terminal = bool(
            fresh
            and task_count > 0
            and completed_count == task_count
            and not active_task_id
            and not implementation_in_progress
            and eligible_ready_count == 0
            and blocked_count == 0
            and external_reserved_count == 0
        )
    return {
        "terminal_quiescent": terminal,
        "task_state_status": "terminal" if terminal else "nonterminal",
        "task_state_path": str(path),
        "task_state_fresh": fresh,
        "task_count": task_count,
        "completed_count": completed_count,
        "active_task_id": active_task_id,
        "implementation_in_progress": implementation_in_progress,
        "eligible_ready_count": eligible_ready_count,
        "blocked_count": blocked_count,
        "external_reserved_count": external_reserved_count,
        "execution_slice_task_ids": list(slice_task_ids),
    }


def _pending_supervisor_generation_fields(
    *,
    status_path: Path,
    observed_at: datetime,
    expected_supervisor_pid: int,
    generation_started_at_epoch_seconds: float,
    startup_grace_seconds: float,
    reason: str,
    observed_supervisor_pid: int | None = None,
    status_age_seconds: float | None = None,
) -> dict[str, object]:
    """Return bounded startup health until this process generation reports."""

    startup_age_seconds = max(
        0.0,
        observed_at.timestamp() - generation_started_at_epoch_seconds,
    )
    within_startup_grace = startup_age_seconds <= startup_grace_seconds
    fields: dict[str, object] = {
        "supervisor_status": "starting" if within_startup_grace else "stale",
        "supervisor_status_generation": "pending",
        "supervisor_status_generation_reason": reason,
        "supervisor_status_path": str(status_path),
        "supervisor_startup_age_seconds": round(startup_age_seconds, 1),
        "supervisor_startup_grace_seconds": startup_grace_seconds,
        "supervisor_within_startup_grace": within_startup_grace,
        "expected_supervisor_pid": expected_supervisor_pid,
        "restart_supervisor": not within_startup_grace,
    }
    if observed_supervisor_pid is not None:
        fields["observed_supervisor_pid"] = observed_supervisor_pid
    if status_age_seconds is not None:
        fields["supervisor_status_age_seconds"] = round(
            status_age_seconds,
            1,
        )
    return fields


def supervisor_status_health_fields(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    stale_seconds: float,
    expected_supervisor_pid: int | None = None,
    generation_started_at_epoch_seconds: float | None = None,
    startup_grace_seconds: float = 0.0,
) -> dict[str, object]:
    """Return generation-bound health for the wrapper supervisor status file."""

    status_path = _inferred_supervisor_status_path(track)
    if status_path is None:
        return {"supervisor_status": "untracked"}
    generation_bound = (
        expected_supervisor_pid is not None
        or generation_started_at_epoch_seconds is not None
    )
    if generation_bound and (
        expected_supervisor_pid is None
        or expected_supervisor_pid <= 0
        or generation_started_at_epoch_seconds is None
    ):
        raise ValueError(
            "supervisor status generation requires a positive PID and spawn epoch"
        )
    observed_at = datetime.now(timezone.utc)
    generation_started_at = (
        float(generation_started_at_epoch_seconds)
        if generation_started_at_epoch_seconds is not None
        else None
    )
    grace = float(startup_grace_seconds)
    if (
        generation_started_at is not None
        and (
            not math.isfinite(generation_started_at)
            or generation_started_at <= 0
            or not math.isfinite(grace)
            or grace < 0
        )
    ):
        raise ValueError("supervisor status generation bounds are invalid")
    payload = _read_json_dict(status_path)
    if not payload:
        if generation_bound:
            return _pending_supervisor_generation_fields(
                status_path=status_path,
                observed_at=observed_at,
                expected_supervisor_pid=int(expected_supervisor_pid),
                generation_started_at_epoch_seconds=float(
                    generation_started_at
                ),
                startup_grace_seconds=grace,
                reason="status_missing",
            )
        return {
            "supervisor_status": "missing",
            "supervisor_status_path": str(status_path),
        }
    updated_at = _parse_status_timestamp(payload.get("updated_at") or payload.get("heartbeat_at"))
    if updated_at is None:
        if generation_bound:
            return _pending_supervisor_generation_fields(
                status_path=status_path,
                observed_at=observed_at,
                expected_supervisor_pid=int(expected_supervisor_pid),
                generation_started_at_epoch_seconds=float(
                    generation_started_at
                ),
                startup_grace_seconds=grace,
                reason="status_timestamp_missing_or_invalid",
            )
        return {
            "supervisor_status": "unknown",
            "supervisor_status_path": str(status_path),
        }
    age_seconds = max(0.0, (observed_at - updated_at).total_seconds())
    observed_pid_value = payload.get("supervisor_pid")
    observed_pid = (
        observed_pid_value
        if isinstance(observed_pid_value, int)
        and not isinstance(observed_pid_value, bool)
        and observed_pid_value > 0
        else None
    )
    if generation_bound and observed_pid != expected_supervisor_pid:
        return _pending_supervisor_generation_fields(
            status_path=status_path,
            observed_at=observed_at,
            expected_supervisor_pid=int(expected_supervisor_pid),
            generation_started_at_epoch_seconds=float(generation_started_at),
            startup_grace_seconds=grace,
            reason=(
                "supervisor_pid_missing"
                if observed_pid is None
                else "supervisor_pid_mismatch"
            ),
            observed_supervisor_pid=observed_pid,
            status_age_seconds=age_seconds,
        )
    if (
        generation_bound
        and updated_at.timestamp() + 1e-6 < float(generation_started_at)
    ):
        return _pending_supervisor_generation_fields(
            status_path=status_path,
            observed_at=observed_at,
            expected_supervisor_pid=int(expected_supervisor_pid),
            generation_started_at_epoch_seconds=float(generation_started_at),
            startup_grace_seconds=grace,
            reason="status_predates_process_generation",
            observed_supervisor_pid=observed_pid,
            status_age_seconds=age_seconds,
        )
    if stale_seconds <= 0 or age_seconds <= stale_seconds:
        return {
            "supervisor_status": "live",
            "supervisor_status_path": str(status_path),
            "supervisor_status_age_seconds": round(age_seconds, 1),
        }

    child_state_path = _relative_or_absolute_path(
        repo_root,
        payload.get("current_status_path") or payload.get("progress_path") or payload.get("state_path"),
    )
    child_state = _read_json_dict(child_state_path)
    active_task_id = str(child_state.get("active_task_id") or "").strip()
    implementation_in_progress = bool(child_state.get("implementation_in_progress"))
    active_child = bool(active_task_id or implementation_in_progress)
    return {
        "supervisor_status": "stale_active" if active_child else "stale",
        "supervisor_status_path": str(status_path),
        "supervisor_status_age_seconds": round(age_seconds, 1),
        "supervisor_active_task_id": active_task_id,
        "supervisor_child_in_progress": implementation_in_progress,
        "restart_supervisor": not active_child,
    }


def format_supervisor_status_fields(fields: Mapping[str, object]) -> str:
    """Return compact supervisor health fields for master heartbeat logs."""

    status = fields.get("supervisor_status")
    if not status or status == "untracked":
        return ""
    parts = [f"supervisor_status={status}"]
    generation = fields.get("supervisor_status_generation")
    if generation:
        parts.append(f"supervisor_status_generation={generation}")
    generation_reason = fields.get("supervisor_status_generation_reason")
    if generation_reason:
        parts.append(
            f"supervisor_status_generation_reason={generation_reason}"
        )
    age = fields.get("supervisor_status_age_seconds")
    if age is not None:
        parts.append(f"supervisor_status_age_seconds={age}")
    active_task_id = fields.get("supervisor_active_task_id")
    if active_task_id:
        parts.append(f"supervisor_active_task_id={active_task_id}")
    startup_age = fields.get("supervisor_startup_age_seconds")
    if startup_age is not None:
        parts.append(f"supervisor_startup_age_seconds={startup_age}")
    if fields.get("supervisor_within_startup_grace"):
        parts.append("supervisor_within_startup_grace=true")
    if fields.get("restart_supervisor"):
        parts.append("restart_supervisor=true")
    return " ".join(parts)


def _persist_plan_bound_process_birth(
    *,
    profile: LifecycleProfile,
    process_identity: ProcessIdentity,
    repo_root: Path,
) -> str:
    """Bind one gated process birth to the active immutable slice before release."""

    from ..control.plan_execution_store import (
        MAX_PLAN_BOUND_WAVE_TRANSFERS,
        PlanBoundExecutionLease,
        PlanBoundProcessBirth,
        ProductionParallelPlanAdapter,
        _load_plan_bound_execution_lease_locked,
        _load_plan_bound_merge_terminal_failure_locked,
        _load_plan_bound_process_birth_chain_locked,
        _load_plan_revision_store_binding_locked,
        _publish_plan_bound_execution_lease_locked,
        _secure_store_active,
        _secure_store_cas,
        _secure_store_continuation,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    def option(name: str) -> str:
        values = _profile_option_values(profile.argv, name)
        if len(values) != 1:
            raise ValueError(f"plan-bound launch requires one exact {name}")
        return values[0]

    revision_cid = option("--plan-bound-revision-cid")
    plan_root_cid = option("--plan-bound-plan-root-cid")
    execution_plan_cid = option("--plan-bound-execution-plan-cid")
    capacity_snapshot_id = option("--plan-bound-capacity-snapshot-id")
    slice_manifest_cid = option("--plan-bound-slice-manifest-cid")
    slice_id = option("--plan-bound-slice-id")
    lane_id = option("--plan-bound-lane-id")
    configuration_root = option("--plan-bound-configuration-root")
    accepted_tree_root = Path(
        option("--plan-bound-accepted-tree-root")
    ).resolve(strict=False)
    if accepted_tree_root != repo_root.resolve():
        raise ValueError("plan-bound birth has a foreign accepted tree")
    task_ids = _profile_option_values(profile.argv, "--execution-slice-task-id")
    task_cids = _profile_option_values(profile.argv, "--execution-slice-task-cid")
    if not task_ids or len(task_ids) != len(task_cids):
        raise ValueError("plan-bound birth has a partial task slice")
    store_path = _resolve_path(
        repo_root,
        Path(option("--plan-revision-store-path")),
    )
    if not _path_within(store_path.resolve(strict=False), repo_root.resolve()):
        raise ValueError("plan-bound birth store escapes accepted tree")
    store = PlanRevisionStore(store_path)
    adapter = ProductionParallelPlanAdapter(store)
    continuation_key = (
        f"plan-bound-process-birth:{revision_cid}:{slice_id}:{lane_id}"
    )
    with store._thread_lock:  # noqa: SLF001 - canonical store transaction
        with store._guard():  # noqa: SLF001 - canonical cross-process guard
            active = _secure_store_active(store)
            if (
                active is None
                or active.revision_cid != revision_cid
                or active.plan_root_cid != plan_root_cid
            ):
                raise ValueError("plan-bound birth lost the active revision fence")
            binding = _load_plan_revision_store_binding_locked(
                store,
                execution_slice_task_ids=task_ids,
                execution_slice_task_cids=task_cids,
            )
            if (
                binding.execution_plan_cid != execution_plan_cid
                or binding.capacity_snapshot_id != capacity_snapshot_id
            ):
                raise ValueError("plan-bound birth observed mixed plan authority")
            reassignment_values = _profile_option_values(
                profile.argv,
                "--plan-bound-reassignment-cid",
            )
            if len(reassignment_values) > 1:
                raise ValueError("plan-bound birth has duplicate reassignment authority")
            execution_slice = adapter._validate_slice_owner_locked(  # noqa: SLF001
                revision_cid=revision_cid,
                slice_manifest_cid=slice_manifest_cid,
                slice_id=slice_id,
                lane_id=lane_id,
                reassignment_cid=(
                    reassignment_values[0] if reassignment_values else ""
                ),
            )
            manifest = _secure_store_cas(store, slice_manifest_cid)
            if (
                execution_slice.task_ids != task_ids
                or execution_slice.task_cids != task_cids
                or manifest.get("configuration_root") != configuration_root
                or manifest.get("source_head")
                != option("--plan-bound-source-head")
                or manifest.get("repository_tree_id")
                != option("--plan-bound-source-tree")
                or manifest.get("task_source_revision")
                != option("--plan-bound-task-source-revision")
            ):
                raise ValueError("plan-bound birth differs from immutable slice")
            if _load_plan_bound_merge_terminal_failure_locked(
                store,
                revision_cid=revision_cid,
                slice_id=slice_id,
            ) is not None:
                raise ValueError(
                    "terminal merge failure forbids another plan-bound birth"
                )

            previous = _load_plan_bound_process_birth_chain_locked(
                store,
                revision_cid=revision_cid,
                slice_id=slice_id,
                lane_id=lane_id,
            )
            prior_birth_cid = ""
            birth_generation = 0
            if previous is not None:
                prior_birth_cid, prior, _prior_chain = previous
                if (
                    prior.plan_root_cid != plan_root_cid
                    or prior.execution_plan_cid != execution_plan_cid
                    or prior.capacity_snapshot_id != capacity_snapshot_id
                    or prior.slice_manifest_cid != slice_manifest_cid
                    or prior.task_ids != tuple(task_ids)
                    or prior.task_cids != tuple(task_cids)
                    or prior.configuration_root != configuration_root
                    or prior.accepted_tree_root != str(accepted_tree_root)
                ):
                    raise ValueError(
                        "prior plan-bound process-birth identity drifted"
                    )
                prior_identity = ProcessIdentity.from_dict(prior.process_birth)
                prior_profile = LifecycleProfile.from_dict(prior.profile)
                if (
                    prior_identity.to_dict() == process_identity.to_dict()
                    and prior_profile.to_dict() == profile.to_dict()
                ):
                    return prior_birth_cid
                prior_state, prior_tree = (
                    _strict_plan_bound_process_fence_observation(
                        prior_profile,
                        prior_identity,
                    )
                )
                prior_tree_is_only_current_gate = (
                    prior_state == "alive"
                    and prior_tree is not None
                    and prior_tree.members == (process_identity,)
                )
                if not (
                    (
                        prior_state == "dead"
                        and prior_tree is not None
                        and not prior_tree.members
                    )
                    or prior_tree_is_only_current_gate
                ):
                    raise ValueError(
                        "prior plan-bound slice birth is not provably fenced"
                    )
                birth_generation = prior.generation + 1
                if birth_generation > MAX_PLAN_BOUND_WAVE_TRANSFERS:
                    raise ValueError(
                        "plan-bound process-birth global budget is exhausted"
                    )

            record = PlanBoundProcessBirth(
                revision_cid=revision_cid,
                plan_root_cid=plan_root_cid,
                execution_plan_cid=execution_plan_cid,
                capacity_snapshot_id=capacity_snapshot_id,
                slice_manifest_cid=slice_manifest_cid,
                slice_id=slice_id,
                lane_id=lane_id,
                task_ids=tuple(task_ids),
                task_cids=tuple(task_cids),
                configuration_root=configuration_root,
                accepted_tree_root=str(accepted_tree_root),
                profile=profile.to_dict(),
                process_birth=process_identity.to_dict(),
                generation=birth_generation,
                global_budget=MAX_PLAN_BOUND_WAVE_TRANSFERS,
                prior_process_birth_cid=prior_birth_cid,
            ).to_dict()
            process_birth_cid = store.put_cas(record)
            if _secure_store_cas(store, process_birth_cid) != record:
                raise ValueError("plan-bound process birth failed CAS round trip")
            continuation = {
                "phase": "committed",
                "operation": "plan_bound_process_birth",
                "revision_cid": revision_cid,
                "slice_id": slice_id,
                "lane_id": lane_id,
                "process_birth_cid": process_birth_cid,
                "generation": birth_generation,
                "global_budget": MAX_PLAN_BOUND_WAVE_TRANSFERS,
            }
            store.put_continuation(
                continuation_key,
                continuation,
            )
            if _secure_store_continuation(store, continuation_key) != continuation:
                raise ValueError(
                    "plan-bound process-birth pointer failed durable round trip"
                )

            raw_assignments = binding.execution_plan.get("assignments")
            if not isinstance(raw_assignments, Sequence) or isinstance(
                raw_assignments,
                (str, bytes, bytearray),
            ):
                raise ValueError("plan-bound execution plan assignments are absent")
            assignments_by_id: dict[str, Mapping[str, Any]] = {}
            for assignment in raw_assignments:
                if not isinstance(assignment, Mapping):
                    raise ValueError("plan-bound compiled assignment is malformed")
                assignment_id = str(assignment.get("task_id") or "")
                if not assignment_id or assignment_id in assignments_by_id:
                    raise ValueError("plan-bound compiled assignments are ambiguous")
                assignments_by_id[assignment_id] = assignment
            compiled_task_bindings: list[dict[str, Any]] = []
            for task_id, task_cid in zip(task_ids, task_cids, strict=True):
                assignment = assignments_by_id.get(task_id)
                if assignment is None:
                    raise ValueError(
                        "plan-bound slice lacks its compiled assignment"
                    )
                compiled_task_bindings.append(
                    {
                        "task_id": task_id,
                        "task_cid": task_cid,
                        "assignment": dict(assignment),
                    }
                )

            prior_execution = _load_plan_bound_execution_lease_locked(
                store,
                revision_cid=revision_cid,
                slice_id=slice_id,
                lane_id=lane_id,
            )
            prior_execution_cid = ""
            execution_generation = 1
            if prior_execution is not None:
                prior_execution_cid, prior_execution_record = prior_execution
                if prior_execution_record.provider_ready:
                    if prior_execution_record.phase in {
                        "proposal_ready",
                        "merge_enqueue_prepared",
                        "merge_enqueue_confirmed",
                    }:
                        # The accepted child may resume only the durable
                        # proposal/merge handoff.  Keep its original provider
                        # effect lease immutable; the new process birth above
                        # is separately bound and recovery never reselects or
                        # redispatches a provider.
                        return process_birth_cid
                    raise ValueError(
                        "prior plan-bound execution reached the provider boundary"
                    )
                if prior_execution_record.daemon_process_birth:
                    from ..merge.worktree_lifecycle import (
                        OwnerLiveness as WorktreeOwnerLiveness,
                    )
                    from ..merge.worktree_lifecycle import (
                        ProcessBirthIdentity as WorktreeProcessBirthIdentity,
                    )
                    from ..merge.worktree_lifecycle import owner_liveness

                    daemon_birth = WorktreeProcessBirthIdentity.from_dict(
                        prior_execution_record.daemon_process_birth
                    )
                    if owner_liveness(daemon_birth) is not WorktreeOwnerLiveness.DEAD:
                        raise ValueError(
                            "prior plan-bound daemon process is not provably dead"
                        )
                execution_generation = prior_execution_record.generation + 1
            execution_lease = PlanBoundExecutionLease(
                revision_cid=revision_cid,
                plan_root_cid=plan_root_cid,
                execution_plan_cid=execution_plan_cid,
                capacity_snapshot_id=capacity_snapshot_id,
                slice_manifest_cid=slice_manifest_cid,
                slice_id=slice_id,
                lane_id=lane_id,
                reassignment_cid=(
                    reassignment_values[0] if reassignment_values else ""
                ),
                task_ids=tuple(task_ids),
                task_cids=tuple(task_cids),
                compiled_task_bindings=tuple(compiled_task_bindings),
                process_birth_cid=process_birth_cid,
                process_birth=process_identity.to_dict(),
                generation=execution_generation,
                phase="reserved",
                prior_execution_lease_cid=prior_execution_cid,
            )
            _publish_plan_bound_execution_lease_locked(
                store,
                execution_lease,
                expected_current_cid=prior_execution_cid,
            )
    return process_birth_cid


def _capture_owned_popen_process_identity(
    process: subprocess.Popen[bytes],
    *,
    profile: LifecycleProfile,
    command: Sequence[str],
    launch_environment: Mapping[str, str],
) -> ProcessIdentity:
    """Capture an exact direct-child birth even after it becomes opaque.

    Credential-bearing supervisor entries deliberately become non-dumpable.
    A fast child can do so between ``Popen`` returning and the parent's
    ``/proc/<pid>/environ`` read.  The parent still has stronger authority
    than a later PID lookup: it created this exact direct child with a new
    session and supplied the complete immutable profile environment.

    Prefer the ordinary marker read.  Permission denial alone selects the
    owned-child fallback, which binds two stable ``/proc/stat`` observations,
    the direct-parent relationship, the dedicated process group/session, the
    supplied profile, and the kernel boot identity.  This direct-child proof
    also covers an accepted entry that clears its inherited marker projection;
    later/adopted PIDs and PID reuse never fall back.
    """

    adapter = LinuxProcessAdapter()
    try:
        return adapter._identity(int(process.pid), profile)  # noqa: SLF001
    except (PermissionError, ProcessIdentityMismatch):
        pass

    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        raise ProcessIdentityMismatch(
            "opaque owned-child birth capture requires Linux /proc"
        )
    if process.poll() is not None:
        raise ProcessIdentityMismatch(
            "owned supervisor exited before process-birth capture"
        )
    first = adapter._stat(int(process.pid))  # noqa: SLF001
    parent_pid, process_group_id, session_id, start_time_ticks = first
    if (
        parent_pid != os.getpid()
        or process_group_id != int(process.pid)
        or session_id != int(process.pid)
    ):
        raise ProcessIdentityMismatch(
            "owned supervisor lacks its direct-child session boundary"
        )
    expected_markers = {
        RUN_ID_ENV: profile.run_id,
        PROFILE_ID_ENV: profile.profile_id,
        TARGET_ID_ENV: profile.target_id,
        REPOSITORY_ROOT_ENV: profile.repository_root,
        STATE_ROOT_ENV: profile.state_root,
        RUN_ROOT_ENV: profile.run_root,
        CONFIGURATION_ROOT_ENV: profile.configuration_root,
    }
    if any(
        launch_environment.get(name) != value
        for name, value in expected_markers.items()
    ):
        raise ProcessIdentityMismatch(
            "owned supervisor launch environment differs from its profile"
        )
    try:
        fencing_epoch = int(launch_environment[FENCING_EPOCH_ENV])
    except (KeyError, ValueError) as exc:
        raise ProcessIdentityMismatch(
            "owned supervisor launch has no lifecycle fence"
        ) from exc
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    if not boot_id:
        raise ProcessIdentityMismatch("kernel boot identity is unavailable")
    executable = shutil.which(
        str(command[0]), path=launch_environment.get("PATH")
    )
    if not executable:
        raise ProcessIdentityMismatch(
            "owned supervisor executable cannot be resolved"
        )
    expected_argv = tuple(str(item) for item in command)
    try:
        observed_argv = adapter._argv(int(process.pid))  # noqa: SLF001
    except PermissionError:
        observed_argv = ()
    if observed_argv and observed_argv != expected_argv:
        raise ProcessIdentityMismatch(
            "owned supervisor command changed before birth capture"
        )
    resolved_executable = str(Path(executable).resolve(strict=True))
    try:
        observed_cwd = str(
            Path(os.readlink(f"/proc/{process.pid}/cwd")).resolve(
                strict=False
            )
        )
        observed_executable = str(
            Path(os.readlink(f"/proc/{process.pid}/exe")).resolve(
                strict=False
            )
        )
    except PermissionError:
        observed_cwd = ""
        observed_executable = ""
    if observed_cwd and observed_cwd != profile.cwd:
        raise ProcessIdentityMismatch(
            "owned supervisor cwd changed before birth capture"
        )
    if observed_executable and observed_executable != resolved_executable:
        raise ProcessIdentityMismatch(
            "owned supervisor executable changed before birth capture"
        )
    identity = ProcessIdentity(
        pid=int(process.pid),
        start_time_ticks=start_time_ticks,
        parent_pid=parent_pid,
        process_group_id=process_group_id,
        session_id=session_id,
        boot_id=boot_id,
        argv=expected_argv,
        cwd=profile.cwd,
        executable=resolved_executable,
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=fencing_epoch,
        configuration_root=profile.configuration_root,
    )
    second = adapter._stat(int(process.pid))  # noqa: SLF001
    if process.poll() is not None or second != first:
        raise ProcessIdentityMismatch(
            "owned supervisor process birth changed during capture"
        )
    return identity


def _fence_failed_owned_process_birth(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 1.0,
) -> bool:
    """Fence one just-created direct child after identity admission fails."""

    try:
        parent, process_group, session, start_time = (
            LinuxProcessAdapter._stat(int(process.pid))  # noqa: SLF001
        )
    except (OSError, ValueError, ProcessLookupError):
        return process.poll() is not None
    if (
        parent != os.getpid()
        or process_group != int(process.pid)
        or session != int(process.pid)
    ):
        return False
    return terminate_pid_tree(
        int(process.pid),
        grace_seconds=max(0.0, grace_seconds),
        freeze_first=True,
        require_gone=True,
        owned_process_group_id=process_group,
        expected_root_start_time_ticks=start_time,
    )


def start_track(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    common_args: Sequence[str],
    python_executable: str = "python3",
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
    output: OutputFn = _default_output,
) -> subprocess.Popen[bytes]:
    """Start one marker-bound supervisor tree and write its PID projection.

    The PID file remains for legacy observability.  Stop/restart decisions use
    the inherited lifecycle markers and exact OS identities attached to the
    returned process, never the PID projection.
    """

    if _configured_board_live_seal_required(common_args, (track,)):
        raise ValueError(CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO)

    resolved = track.resolve(repo_root)
    child_command = (
        [python_executable, "-m", resolved.module_name, *resolved.extra_args]
        if resolved.module_name
        else [python_executable, str(resolved.script_path), *common_args, *resolved.extra_args]
    )
    plan_bound_dispatch = "--plan-bound-dispatch" in resolved.extra_args
    gate_read_fd: int | None = None
    gate_write_fd: int | None = None
    recovery_authorization_cid = ""
    retained_interpreter: RetainedControlPlaneInterpreter | None = None
    native_dependency: AgentSupervisorNativeDependencyLaunch | None = None
    system_dependency_directories = ""
    accepted_tree_root = _canonical_accepted_tree_root(Path(repo_root))
    command = child_command
    if plan_bound_dispatch:
        accepted_roots = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-accepted-tree-root",
        )
        configuration_roots = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-configuration-root",
        )
        store_paths = _profile_option_values(
            resolved.extra_args,
            "--plan-revision-store-path",
        )
        source_heads = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-source-head",
        )
        source_trees = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-source-tree",
        )
        revision_cids = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-revision-cid",
        )
        slice_ids = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-slice-id",
        )
        lane_ids = _profile_option_values(
            resolved.extra_args,
            "--plan-bound-lane-id",
        )
        state_dirs = _profile_option_values(
            resolved.extra_args,
            "--state-dir",
        )
        state_prefixes = _profile_option_values(
            resolved.extra_args,
            "--state-prefix",
        )
        launch_args = (*common_args, *resolved.extra_args)
        worktree_roots = _profile_option_values(
            launch_args,
            "--worktree-root",
        )
        merge_queue_roots = _profile_option_values(
            launch_args,
            "--merge-queue-dir",
        )
        canonical_repo_root = accepted_tree_root
        if (
            resolved.module_name
            or len(accepted_roots) != 1
            or Path(accepted_roots[0]) != canonical_repo_root
            or Path(python_executable).resolve(strict=False)
            != Path(sys.executable).resolve(strict=False)
            or resolved.script_path
            != accepted_tree_root / PLAN_BOUND_ACCEPTED_ENTRY_PATH
            or len(configuration_roots) != 1
            or not configuration_roots[0]
            or len(store_paths) != 1
            or len(source_heads) != 1
            or len(source_trees) != 1
            or len(revision_cids) != 1
            or len(slice_ids) != 1
            or len(lane_ids) != 1
            or len(state_dirs) != 1
            or len(state_prefixes) != 1
        ):
            raise ValueError(
                "plan-bound dispatch is not pinned to the accepted tree entry"
            )
        if accepted_control_plane_pin is None:
            raise ValueError(
                "plan-bound dispatch requires a sealed accepted control plane"
            )
        verify_agent_implementation_sealed_control_plane(
            accepted_control_plane_pin,
            accepted_control_plane_descriptor,
        )
        native_dependency, system_dependency_directories = (
            admit_sealed_native_dependency_environment(os.environ)
        )
        retained_interpreter = retain_control_plane_interpreter(
            python_executable
        )
        if (
            accepted_control_plane_pin.source_head != source_heads[0]
            or accepted_control_plane_pin.source_tree != source_trees[0]
        ):
            raise ValueError(
                "plan-bound slice differs from the accepted control-plane generation"
            )
        plan_store = _resolve_path(repo_root, Path(store_paths[0]))
        state_dir = _resolve_path(repo_root, Path(state_dirs[0]))
        if state_dir.parent != plan_store.parent:
            raise ValueError(
                "plan-bound state and store do not share the configured state root"
            )
        # Validate the lexical authority paths before PlanRevisionStore may
        # resolve or create anything.  In particular, a dangling/intermediate
        # symlink supplied as the store path must not redirect the first
        # recovery read or create a directory outside the configured root.
        for authority_path in (
            resolved.script_path,
            resolved.log_path,
            resolved.supervisor_pid_path,
            resolved.daemon_pid_path,
            plan_store,
        ):
            _lexical_contained_path(
                canonical_repo_root,
                authority_path,
                require_regular=authority_path == resolved.script_path,
            )
        for runtime_path in (
            resolved.log_path,
            resolved.supervisor_pid_path,
            resolved.daemon_pid_path,
        ):
            if runtime_path.parent != state_dir:
                raise ValueError(
                    "plan-bound runtime projection escapes its configured lane state"
                )
        # Reject a preplaced PID projection before PlanRevisionStore reads or
        # Git identity probes can cross a subprocess boundary.  The later
        # O_EXCL reservation repeats this under its update lock to close the
        # check-to-create race.
        with serialized_lock_update(resolved.supervisor_pid_path):
            _require_absent_pid_projection(resolved.supervisor_pid_path)
        from ..control.plan_execution_store import ProductionParallelPlanAdapter
        from ..task_sources.plan_revision_store import PlanRevisionStore

        plan_adapter = ProductionParallelPlanAdapter(
            PlanRevisionStore(plan_store)
        )
        current_execution = plan_adapter.load_execution_lease(
            revision_cid=revision_cids[0],
            slice_id=slice_ids[0],
            lane_id=lane_ids[0],
        )
        recovery_phase = (
            current_execution is not None
            and current_execution[1].phase
            in {
                "proposal_ready",
                "merge_enqueue_prepared",
                "merge_enqueue_confirmed",
            }
        )
        repository_head, repository_tree = _plan_bound_repository_identity(
            accepted_tree_root
        )
        recovery_decision = None
        recovery_runtime_roots: tuple[Path, ...] = ()
        recovery_owner_bound_artifacts: tuple[Path, ...] = ()
        recovery_artifacts: tuple[Mapping[str, Any], ...] = ()
        if recovery_phase:
            if len(worktree_roots) != 1 or len(merge_queue_roots) != 1:
                raise ValueError(
                    "plan-bound recovery lacks exact configured runtime roots"
                )
            worktree_root = _resolve_path(
                canonical_repo_root,
                Path(worktree_roots[0]),
            )
            merge_queue_root = _resolve_path(
                canonical_repo_root,
                Path(merge_queue_roots[0]),
            )
            recovery_runtime_roots = (
                plan_store.parent,
                worktree_root,
                merge_queue_root,
            )
            recovery_runtime_bindings = plan_adapter.recovery_runtime_bindings(
                revision_cid=revision_cids[0],
                slice_manifest_cid=current_execution[1].slice_manifest_cid,
            )
            workspace_paths = tuple(
                _resolve_path(canonical_repo_root, Path(path))
                for path in plan_adapter.recovery_workspace_paths(
                    revision_cid=revision_cids[0],
                    slice_manifest_cid=current_execution[1].slice_manifest_cid,
                )
            )
            implementation_lock = state_dir / "implementation.lock"
            launch_owned_paths = (
                resolved.log_path,
                resolved.supervisor_pid_path,
            )
            recovery_owner_bound_artifacts = (
                implementation_lock,
                *workspace_paths,
                resolved.supervisor_pid_path.with_name(
                    f".{resolved.supervisor_pid_path.name}.update.lock"
                ),
                *launch_owned_paths,
            )
            recovery_artifacts = _snapshot_plan_bound_recovery_artifacts(
                root=canonical_repo_root,
                runtime_roots=(
                    plan_store.parent,
                    worktree_root,
                    merge_queue_root,
                ),
                owner_bound_artifacts=recovery_owner_bound_artifacts,
                runtime_bindings=recovery_runtime_bindings,
                state_dir=state_dir,
                state_prefix=state_prefixes[0],
            )
            recovery_authorization_cid, recovery_decision = (
                plan_adapter.authorize_recovery_launch(
                    revision_cid=revision_cids[0],
                    slice_id=slice_ids[0],
                    lane_id=lane_ids[0],
                    source_head=source_heads[0],
                    source_tree=source_trees[0],
                    repository_head=repository_head,
                    repository_tree=repository_tree,
                    runtime_artifacts=recovery_artifacts,
                    launch_artifact_paths=tuple(
                        sorted(
                            path.relative_to(canonical_repo_root).as_posix()
                            for path in launch_owned_paths
                        )
                    ),
                )
            )
        _validate_plan_bound_accepted_tree(
            accepted_tree_root=accepted_tree_root,
            source_head=source_heads[0],
            source_tree=source_trees[0],
            control_plane_pin=accepted_control_plane_pin,
            recovery_repository_head=(
                "" if recovery_decision is None else recovery_decision.repository_head
            ),
            recovery_repository_tree=(
                "" if recovery_decision is None else recovery_decision.repository_tree
            ),
            recovery_runtime_roots=recovery_runtime_roots,
            recovery_owner_bound_artifacts=(
                recovery_owner_bound_artifacts
            ),
            recovery_artifacts=recovery_artifacts,
        )
        # The accepted-tree gate process cannot exec the requested supervisor
        # until the parent captures its exact lifecycle birth and explicitly
        # releases one byte.  Thus even a /proc identity failure cannot race a
        # daemon preclaim or provider effect.
        gate_read_fd, gate_write_fd = os.pipe()
        supervisor_argv = [
            *common_args,
            *resolved.extra_args,
            "--accepted-control-plane-pin-json",
            accepted_control_plane_pin_json(accepted_control_plane_pin),
            "--accepted-control-plane-fd",
            str(accepted_control_plane_descriptor),
        ]
        child_command = build_sealed_control_plane_module_command(
            python_executable=retained_interpreter.argv0,
            pin=accepted_control_plane_pin,
            descriptor=accepted_control_plane_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=supervisor_argv,
            retained_interpreter=retained_interpreter,
            native_dependency_launch=native_dependency,
            accepted_native_authorization_id=(
                native_dependency.accepted_authorization_id
            ),
            system_dependency_directories_json=(
                system_dependency_directories
            ),
        )
        gate_argv = [
            PLAN_BOUND_LAUNCH_GATE_MARKER,
            str(gate_read_fd),
            str(accepted_tree_root),
            accepted_control_plane_pin_json(accepted_control_plane_pin),
            str(accepted_control_plane_descriptor),
            recovery_authorization_cid or "-",
            str(retained_interpreter.descriptor),
            retained_interpreter.argv0,
            retained_interpreter.sha256,
            "--",
            *child_command,
        ]
        command = build_sealed_control_plane_module_command(
            python_executable=retained_interpreter.argv0,
            pin=accepted_control_plane_pin,
            descriptor=accepted_control_plane_descriptor,
            module_name=PLAN_BOUND_LAUNCH_GATE_MODULE,
            argv=gate_argv,
            retained_interpreter=retained_interpreter,
            native_dependency_launch=native_dependency,
            accepted_native_authorization_id=(
                native_dependency.accepted_authorization_id
            ),
            system_dependency_directories_json=(
                system_dependency_directories
            ),
        )
    resolved.log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.supervisor_pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_reservation_fd: int | None = None
    pid_reservation_identity: tuple[int, int] | None = None
    if plan_bound_dispatch:
        (
            pid_reservation_fd,
            pid_reservation_identity,
        ) = _reserve_owned_pid_projection(resolved.supervisor_pid_path)
    configuration_root = "sha256:" + hashlib.sha256(
        json.dumps(
            command, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    state_root = resolved.supervisor_pid_path.parent.resolve(strict=False)
    run_root = state_root / "lifecycle-runs" / resolved.name
    status_path = _inferred_supervisor_status_path(resolved)
    profile_environment_values = dict(
        _plan_bound_profile_environment(os.environ)
        if plan_bound_dispatch
        else ()
    )
    profile_environment_values.update(
        _trusted_duckdb_profile_environment(
            os.environ,
            repository_root=repo_root,
        )
    )
    profile_environment = tuple(sorted(profile_environment_values.items()))
    profile = LifecycleProfile(
        target_id=f"supervisor-track:{resolved.name}",
        run_id=(
            "multi-supervisor:"
            + hashlib.sha256(
                f"{repo_root.resolve()}:{resolved.name}".encode()
            ).hexdigest()
        ),
        configuration_root=configuration_root,
        repository_root=str(repo_root.resolve()),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=tuple(command),
        cwd=str(repo_root.resolve()),
        environment=profile_environment,
        health_path=(
            str(status_path.resolve(strict=False))
            if status_path is not None
            and _path_within(status_path.resolve(strict=False), state_root)
            else ""
        ),
    )
    try:
        out_handle = resolved.log_path.open("ab")
    except BaseException:
        if pid_reservation_fd is not None:
            os.close(pid_reservation_fd)
        if pid_reservation_identity is not None:
            _discard_reserved_pid_projection(
                resolved.supervisor_pid_path,
                pid_reservation_identity,
            )
        raise
    launch_environment = profile.launch_environment(0)
    # Resource policy is non-secret and follows the host's supervisor handoff.
    # An explicit sealed profile value takes precedence over ambient defaults.
    for name in HASH_RESOURCE_ENV_NAMES:
        if name not in launch_environment and str(os.environ.get(name, "") or "").strip():
            launch_environment[name] = os.environ[name]
    # Both ordinary configured-board tracks and plan-bound tracks need the
    # live owner socket plus sealed broker descriptor. Lifecycle profiles are
    # positive projections, so ambient inheritance cannot supply these later.
    launch_environment.update(
        {
            name: os.environ[name]
            for name in (
                STATE_OWNER_SOCKET_ENV,
                *TRUSTED_STATE_GRANT_BROKER_ENV_NAMES,
            )
            if str(os.environ.get(name, "") or "").strip()
        }
    )
    if native_dependency is None:
        admitted_native = optional_active_sealed_native_dependency(os.environ)
        if admitted_native is not None:
            native_dependency, system_dependency_directories = admitted_native
    native_pass_fds: tuple[int, ...] = ()
    if native_dependency is not None:
        launch_environment.update(
            sealed_native_dependency_environment(
                native_dependency,
                system_dependency_directories_json=(
                    system_dependency_directories
                ),
            )
        )
        native_pass_fds = (native_dependency.descriptor.descriptor,)
    if plan_bound_dispatch:
        # Isolated absolute-script launch bootstraps only its own accepted
        # repository root.  Build a positive environment in the parent before
        # the interpreter is born; clearing loader knobs in the bootstrap
        # would be too late for LD_PRELOAD.  The sealed native dependency's
        # DT_NEEDED resolution is intentionally bounded to the host's default
        # system ABI, not caller-provided loader/search configuration.
        # HOME, Python user-base, and cache bindings enter this profile only
        # through _trusted_duckdb_profile_environment, which derives them from
        # the independently admitted marker. They are not ambient allowlist
        # members, but they are valid sealed profile fields at this boundary.
        route_names = {
            *_PLAN_BOUND_PROFILE_ENV_NAMES,
            "HOME",
            TRUSTED_PYTHON_USER_BASE_ENV,
            *TRUSTED_RUNTIME_CACHE_ENV_NAMES,
        }
        explicit_profile = dict(profile.environment)
        disallowed_profile_names = set(explicit_profile) - route_names
        if disallowed_profile_names:
            raise ValueError(
                "plan-bound lifecycle profile contains non-route environment"
            )
        launch_environment = _plan_bound_positive_child_environment(
            launch_environment
        )
        launch_environment.pop(TRUSTED_PYTHON_USER_BASE_ENV, None)
        assert native_dependency is not None
        launch_environment.update(
            sealed_native_dependency_environment(
                native_dependency,
                system_dependency_directories_json=(
                    system_dependency_directories
                ),
            )
        )
    from .process_security import (
        STATE_AUTHORITY_PARENT_LOSS_TERMINATE,
        prepare_state_authority_child_handoff,
        state_authority_credentials_present,
    )

    try:
        exact_birth_required = state_authority_credentials_present(
            launch_environment
        )
        authority_handoff = prepare_state_authority_child_handoff(
            launch_environment,
            parent_loss_policy=STATE_AUTHORITY_PARENT_LOSS_TERMINATE,
        )
        try:
            cleanup_directory_anchor = (
                _open_durable_cleanup_directory_anchor(run_root)
            )
        except BaseException:
            authority_handoff.close()
            raise
        authority_descriptors = authority_handoff.pass_fds
    except BaseException:
        out_handle.close()
        if gate_read_fd is not None:
            os.close(gate_read_fd)
        if gate_write_fd is not None:
            os.close(gate_write_fd)
        if pid_reservation_fd is not None:
            os.close(pid_reservation_fd)
        if pid_reservation_identity is not None:
            _discard_reserved_pid_projection(
                resolved.supervisor_pid_path,
                pid_reservation_identity,
            )
        raise
    process: subprocess.Popen[bytes] | None = None
    try:
        try:
            process = subprocess.Popen(
                command,
                executable=(
                    retained_interpreter.executable_path
                    if retained_interpreter is not None
                    else None
                ),
                cwd=repo_root,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=tuple(
                    sorted(
                        {
                            *authority_descriptors,
                            *native_pass_fds,
                            *(
                                (
                                    gate_read_fd,
                                    accepted_control_plane_descriptor,
                                    retained_interpreter.descriptor,
                                )
                                if plan_bound_dispatch and gate_read_fd is not None
                                else ()
                            ),
                        }
                    )
                ),
            )
            process._agent_supervisor_cleanup_directory_anchor = (
                cleanup_directory_anchor
            )
            process._agent_supervisor_cleanup_directory_anchor_required = True
            if not plan_bound_dispatch:
                authority_handoff.deliver(process)
        except BaseException:
            authority_handoff.close()
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            try:
                os.close(cleanup_directory_anchor.descriptor)
            except OSError:
                pass
            if gate_read_fd is not None:
                os.close(gate_read_fd)
            if gate_write_fd is not None:
                os.close(gate_write_fd)
            if pid_reservation_fd is not None:
                os.close(pid_reservation_fd)
            if pid_reservation_identity is not None:
                _discard_reserved_pid_projection(
                    resolved.supervisor_pid_path,
                    pid_reservation_identity,
                )
            raise
    finally:
        out_handle.close()
    if process is None:
        raise AssertionError("managed process launch returned no child")
    if gate_read_fd is not None:
        os.close(gate_read_fd)
    # Popen is only an observation handle.  The immutable profile is what lets
    # stop/restart rediscover children that have detached or been reparented.
    process._agent_supervisor_lifecycle_profile = profile
    if plan_bound_dispatch:
        if gate_write_fd is None:
            raise AssertionError("plan-bound launch gate was not created")
        try:
            # Capture the exact process birth while the accepted-tree gate is
            # still blocking the requested supervisor command.
            process_identity = LinuxProcessAdapter()._identity(  # noqa: SLF001
                int(process.pid), profile
            )
            if not isinstance(process_identity, ProcessIdentity):
                raise ProcessIdentityMismatch(
                    "plan-bound launch returned no typed process identity"
                )
            process._agent_supervisor_process_identity = process_identity
            birth_cid = _persist_plan_bound_process_birth(
                profile=profile,
                process_identity=process_identity,
                repo_root=Path(repo_root).resolve(),
            )
            process._agent_supervisor_process_birth_cid = birth_cid
            if (
                pid_reservation_fd is None
                or pid_reservation_identity is None
            ):
                raise AssertionError(
                    "plan-bound PID projection was not reserved"
                )
            _publish_reserved_pid_projection(
                resolved.supervisor_pid_path,
                pid_reservation_fd,
                pid_reservation_identity,
                int(process.pid),
            )
            os.close(pid_reservation_fd)
            pid_reservation_fd = None
            if os.write(
                gate_write_fd, PLAN_BOUND_LAUNCH_GATE_SUCCESS
            ) != len(PLAN_BOUND_LAUNCH_GATE_SUCCESS):
                raise OSError("plan-bound launch gate release was incomplete")
            # The launch gate deliberately carries no authority.  Its final
            # exec preserves PID/start identity but resets dumpability; only
            # the sealed target redeems after that exec hardens itself.
            authority_handoff.deliver(
                process,
                expected_executable_descriptor=(
                    retained_interpreter.descriptor
                ),
                expected_argv=child_command,
            )
        except Exception as exc:
            try:
                os.close(gate_write_fd)
            except OSError:
                pass
            gate_write_fd = None
            all_trees_fenced = _fence_unreleased_plan_bound_process(process)
            if pid_reservation_fd is not None:
                try:
                    os.close(pid_reservation_fd)
                except OSError:
                    pass
                pid_reservation_fd = None
            if pid_reservation_identity is not None:
                _discard_reserved_pid_projection(
                    resolved.supervisor_pid_path,
                    pid_reservation_identity,
                )
            try:
                os.close(cleanup_directory_anchor.descriptor)
            except OSError:
                pass
            raise PlanBoundProcessBirthError(
                "plan-bound process birth capture failed; launch remained gated",
                pid=int(process.pid),
                profile=profile,
                all_trees_fenced=all_trees_fenced,
            ) from exc
        finally:
            if gate_write_fd is not None:
                os.close(gate_write_fd)
    else:
        try:
            process_identity = _capture_owned_popen_process_identity(
                process,
                profile=profile,
                command=command,
                launch_environment=launch_environment,
            )
        except (
            OSError,
            UnicodeError,
            ValueError,
            ProcessIdentityMismatch,
        ) as exc:
            if exact_birth_required:
                fenced = _fence_failed_owned_process_birth(process)
                try:
                    process.wait(timeout=2.0)
                except (ChildProcessError, OSError, subprocess.TimeoutExpired):
                    pass
                if not fenced:
                    try:
                        os.close(cleanup_directory_anchor.descriptor)
                    except OSError:
                        pass
                    raise ProcessIdentityMismatch(
                        "credential-bearing supervisor birth failed and its "
                        "process tree could not be fenced"
                    ) from exc
                try:
                    os.close(cleanup_directory_anchor.descriptor)
                except OSError:
                    pass
                raise ProcessIdentityMismatch(
                    "credential-bearing supervisor birth identity is unavailable"
                ) from exc
            # Credential-free legacy tracks retain best-effort observability.
            process_identity = None
        process._agent_supervisor_process_identity = process_identity
        resolved.supervisor_pid_path.write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
    if plan_bound_dispatch:
        if not isinstance(process_identity, ProcessIdentity):
            raise ProcessIdentityMismatch(
                "plan-bound managed daemon fence lacks the lane birth"
            )
        try:
            process._agent_supervisor_managed_daemon_kernel_fence = (
                _managed_daemon_kernel_fence_for_track(
                    resolved,
                    repo_root=repo_root,
                    launch_argv=(*common_args, *resolved.extra_args),
                    process_identity=process_identity,
                )
            )
        except Exception as exc:
            fenced, _members = _terminate_managed_process(
                process,
                grace_seconds=1.0,
            )
            if fenced:
                _remove_stale_pid_marker_if_unchanged(
                    resolved.supervisor_pid_path,
                    int(process.pid),
                )
            raise PlanBoundProcessBirthError(
                "plan-bound daemon kernel fence construction failed",
                pid=int(process.pid),
                profile=profile,
                all_trees_fenced=fenced,
            ) from exc
    _emit(
        output,
        f"started {resolved.name} supervisor pid={process.pid} script={resolved.script_path} log={resolved.log_path}",
    )
    if retained_interpreter is not None:
        os.close(retained_interpreter.descriptor)
    return process


def _fence_unreleased_plan_bound_process(
    process: subprocess.Popen[bytes],
) -> bool:
    """Fence the exact owned session/group, including a post-gate fork."""

    identity = getattr(
        process,
        "_agent_supervisor_process_identity",
        None,
    )
    try:
        if not isinstance(identity, ProcessIdentity):
            fenced = _fence_failed_owned_process_birth(process)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                return False
            return bool(fenced and process.poll() is not None)
        if (
            identity.pid != int(process.pid)
            or identity.process_group_id != int(process.pid)
            or identity.session_id != int(process.pid)
        ):
            return False
        fenced = terminate_pid_tree(
            int(process.pid),
            grace_seconds=1.0,
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=int(process.pid),
            expected_root_start_time_ticks=(
                identity.start_time_ticks
            ),
        )
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return False
        return bool(fenced and process.poll() is not None)
    except (OSError, ProcessIdentityMismatch, RuntimeError):
        return False


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _canonical_accepted_tree_root(path: Path) -> Path:
    """Reject aliased or symlinked accepted-tree roots lexically."""

    root = Path(path)
    if not root.is_absolute() or Path(os.path.abspath(root)) != root:
        raise ValueError("accepted tree root is not lexical absolute")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise ValueError(
                f"cannot lstat accepted tree component: {current}"
            ) from exc
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ValueError(
                f"accepted tree component is not a real directory: {current}"
            )
    if root.resolve(strict=True) != root:
        raise ValueError("accepted tree root is not canonical")
    return root


def _lexical_contained_path(
    root: Path,
    path: Path,
    *,
    require_regular: bool = False,
) -> Path:
    """Validate containment and reject every symlinked existing component."""

    candidate = Path(path)
    if not candidate.is_absolute() or Path(os.path.abspath(candidate)) != candidate:
        raise ValueError(f"plan-bound path is not lexical absolute: {candidate}")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"plan-bound path escapes accepted tree: {candidate}") from exc
    current = root
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            observed = os.lstat(current)
        except FileNotFoundError:
            # Missing descendants are safe to create only after all existing
            # lexical parents have been checked.
            break
        except OSError as exc:
            raise ValueError(f"cannot lstat plan-bound path: {current}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"plan-bound path contains a symbolic link: {current}")
        final = index == len(relative.parts) - 1
        if final and require_regular:
            if not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1:
                raise ValueError(
                    f"plan-bound entry is not a single-link regular file: {current}"
                )
        elif not final and not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"plan-bound parent is not a directory: {current}")
    if require_regular and not candidate.exists():
        raise ValueError(f"plan-bound entry is absent: {candidate}")
    return candidate


def _plan_bound_git_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"LANG", "LC_ALL", "LC_CTYPE", "TZ"}
    }
    environment.update(
        {
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _plan_bound_git(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            f"--work-tree={root}",
            *args,
        ],
        cwd=root,
        env=_plan_bound_git_environment(),
        input=input_bytes,
        text=input_bytes is None,
        capture_output=True,
        check=False,
        timeout=30.0,
    )


def _plan_bound_repository_identity(root: Path) -> tuple[str, str]:
    """Read one exact current Git HEAD/tree pair with the sanitized client."""

    head = _plan_bound_git(root, "rev-parse", "HEAD")
    tree = _plan_bound_git(root, "rev-parse", "HEAD^{tree}")
    head_id = str(head.stdout).strip()
    tree_id = str(tree.stdout).strip()
    if (
        head.returncode != 0
        or tree.returncode != 0
        or re.fullmatch(r"[0-9a-f]{40}", head_id) is None
        or re.fullmatch(r"[0-9a-f]{40}", tree_id) is None
    ):
        raise ValueError("plan-bound repository identity is unavailable")
    return head_id, tree_id


def _plan_bound_recovery_artifact_evidence(
    root: Path,
    artifact: Path,
    *,
    workspace: bool,
) -> dict[str, Any]:
    """Return stable owner/mode/content evidence for one exact runtime path."""

    relative = artifact.relative_to(root).as_posix()
    if workspace:
        observed = os.lstat(artifact)
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or int(observed.st_uid) != os.geteuid()
            or bool(stat.S_IMODE(observed.st_mode) & 0o7000)
            or bool(stat.S_IMODE(observed.st_mode) & 0o022)
        ):
            raise ValueError("recovery workspace directory custody is unsafe")
        marker = artifact / ".git"
        marker_bytes, marker_evidence = _read_stable_regular_bytes(
            marker,
            max_bytes=16_384,
        )
        if (
            marker_bytes is None
            or int(marker_evidence["uid"]) != os.geteuid()
            or int(marker_evidence["link_count"]) != 1
            or stat.S_IMODE(int(marker_evidence["mode"])) != 0o600
        ):
            raise ValueError("recovery workspace Git marker custody is unsafe")
        try:
            marker_text = marker_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("recovery workspace Git marker is not text") from exc
        if not marker_text.startswith("gitdir: "):
            raise ValueError("recovery workspace Git marker is malformed")
        git_dir = Path(marker_text[8:])
        if not git_dir.is_absolute():
            git_dir = artifact / git_dir
        git_dir = Path(os.path.abspath(git_dir))
        canonical_git_root = _lexical_contained_path(root, root / ".git")
        if not git_dir.is_relative_to(canonical_git_root / "worktrees"):
            raise ValueError("recovery workspace Git custody escapes the repository")
        _lexical_contained_path(root, git_dir)
        git_custody: list[bytes] = []
        current_git_path = canonical_git_root
        for part in git_dir.relative_to(canonical_git_root).parts:
            try:
                git_stat = os.lstat(current_git_path)
            except OSError as exc:
                raise ValueError(
                    "recovery workspace Git custody is unreadable"
                ) from exc
            if (
                not stat.S_ISDIR(git_stat.st_mode)
                or stat.S_ISLNK(git_stat.st_mode)
                or int(git_stat.st_uid) != os.geteuid()
                or bool(stat.S_IMODE(git_stat.st_mode) & 0o7022)
            ):
                raise ValueError("recovery workspace Git custody is unsafe")
            git_custody.append(
                (
                    f"{current_git_path.relative_to(root).as_posix()}:"
                    f"{stat.S_IMODE(git_stat.st_mode)}:{git_stat.st_uid}:"
                    f"{git_stat.st_nlink}"
                ).encode()
            )
            current_git_path /= part
        git_stat = os.lstat(git_dir)
        if (
            not stat.S_ISDIR(git_stat.st_mode)
            or stat.S_ISLNK(git_stat.st_mode)
            or int(git_stat.st_uid) != os.geteuid()
            or bool(stat.S_IMODE(git_stat.st_mode) & 0o7022)
        ):
            raise ValueError("recovery workspace Git custody is unsafe")
        git_custody.append(
            (
                f"{git_dir.relative_to(root).as_posix()}:"
                f"{stat.S_IMODE(git_stat.st_mode)}:{git_stat.st_uid}:"
                f"{git_stat.st_nlink}"
            ).encode()
        )
        top = _plan_bound_git(artifact, "rev-parse", "--show-toplevel")
        common = _plan_bound_git(artifact, "rev-parse", "--git-common-dir")
        head = _plan_bound_git(artifact, "rev-parse", "HEAD")
        if (
            top.returncode != 0
            or Path(str(top.stdout).strip()) != artifact
            or common.returncode != 0
            or Path(str(common.stdout).strip()).resolve(strict=True)
            != canonical_git_root.resolve(strict=True)
            or head.returncode != 0
            or re.fullmatch(r"[0-9a-f]{40}", str(head.stdout).strip()) is None
        ):
            raise ValueError("recovery workspace lost canonical Git custody")
        digest_payload = b"\0".join(
            (
                marker_bytes,
                (
                    f"marker:{stat.S_IMODE(int(marker_evidence['mode']))}:"
                    f"{int(marker_evidence['uid'])}:"
                    f"{int(marker_evidence['link_count'])}"
                ).encode("ascii"),
                str(head.stdout).strip().encode("ascii"),
                git_dir.relative_to(root).as_posix().encode("utf-8"),
                *git_custody,
            )
        )
        return {
            "path": relative,
            "kind": "workspace",
            "sha256": "sha256:" + hashlib.sha256(digest_payload).hexdigest(),
            "mode": stat.S_IMODE(observed.st_mode),
            "uid": int(observed.st_uid),
            "nlink": int(observed.st_nlink),
            "size": int(observed.st_size),
        }

    payload, evidence = _read_stable_regular_bytes(
        artifact,
        max_bytes=134_217_728,
    )
    if (
        payload is None
        or int(evidence["uid"]) != os.geteuid()
        or int(evidence["link_count"]) != 1
        or bool(stat.S_IMODE(int(evidence["mode"])) & 0o111)
    ):
        raise ValueError("recovery runtime artifact custody is unsafe")
    return {
        "path": relative,
        "kind": "file",
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "mode": stat.S_IMODE(int(evidence["mode"])),
        "uid": int(evidence["uid"]),
        "nlink": int(evidence["link_count"]),
        "size": int(evidence["size"]),
    }


def _plan_bound_safe_store_filename(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    cleaned = "".join(
        character
        if character.isalnum() or character in "._-@"
        else "_"
        for character in value
    )[:96]
    return f"{cleaned}.{digest[:16]}"


def _validate_plan_bound_store_projection(
    store_root: Path,
    artifact: Path,
) -> None:
    """Accept only exact canonical PlanRevisionStore projection files."""

    relative = artifact.relative_to(store_root)
    parts = relative.parts
    try:
        projection_stat = os.lstat(artifact)
    except OSError as exc:
        raise ValueError("plan store projection custody is unreadable") from exc
    if parts == (".plan-revision-store.lock",):
        payload, evidence = _read_stable_regular_bytes(artifact, max_bytes=0)
        if (
            payload != b""
            or int(evidence["uid"]) != os.geteuid()
            or int(evidence["link_count"]) != 1
            or bool(stat.S_IMODE(int(evidence["mode"])) & 0o111)
        ):
            raise ValueError("plan store lock projection is unsafe")
        return
    if (
        not stat.S_ISREG(projection_stat.st_mode)
        or stat.S_ISLNK(projection_stat.st_mode)
        or int(projection_stat.st_uid) != os.geteuid()
        or int(projection_stat.st_nlink) != 1
        or stat.S_IMODE(projection_stat.st_mode) != 0o600
    ):
        raise ValueError("plan store projection custody is unsafe")

    from ..task_sources.plan_revision_store import (
        PLAN_REVISION_ACTIVE_SCHEMA,
        PLAN_REVISION_APPLY_RECEIPT_SCHEMA,
        PLAN_REVISION_CONTINUATION_SCHEMA,
        PLAN_REVISION_EVENT_SCHEMA,
        PLAN_REVISION_INDEX_SCHEMA,
        PLAN_REVISION_INTENT_SCHEMA,
        PLAN_REVISION_STORE_SCHEMA,
        PLAN_REVISION_SUPERSESSION_SCHEMA,
        PlanRevisionActiveProjection,
        PlanRevisionIntent,
    )

    if parts == ("active.json",):
        payload, _evidence = _read_stable_regular_json(artifact)
        if payload is None or payload.get("schema") != PLAN_REVISION_ACTIVE_SCHEMA:
            raise ValueError("plan store active projection is malformed")
        active = PlanRevisionActiveProjection.from_dict(payload)
        if active.to_dict() != payload:
            raise ValueError("plan store active projection normalized")
        return
    if parts == ("index.json",):
        payload, _evidence = _read_stable_regular_json(artifact)
        if (
            payload is None
            or set(payload) != {
                "schema",
                "revisions",
                "deltas",
                "latest_revision_cid",
                "latest_intent_cid",
            }
            or payload.get("schema") != PLAN_REVISION_INDEX_SCHEMA
            or not isinstance(payload.get("revisions"), list)
            or not isinstance(payload.get("deltas"), list)
        ):
            raise ValueError("plan store index projection is malformed")
        return
    if parts in {("events.jsonl",), ("supersessions.jsonl",)}:
        payload, evidence = _read_stable_regular_bytes(artifact, max_bytes=8_388_608)
        if payload is None or int(evidence["uid"]) != os.geteuid():
            raise ValueError("plan store append-only projection is unsafe")
        expected_schema = (
            PLAN_REVISION_EVENT_SCHEMA
            if parts == ("events.jsonl",)
            else PLAN_REVISION_SUPERSESSION_SCHEMA
        )
        for raw_line in payload.splitlines():
            if not raw_line:
                continue
            try:
                record = json.loads(
                    raw_line,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("plan store append-only record is malformed") from exc
            if not isinstance(record, Mapping) or record.get("schema") != expected_schema:
                raise ValueError("plan store append-only record schema is mixed")
        return
    if len(parts) == 2 and parts[0] == "cas":
        payload, _evidence = _read_stable_regular_json(artifact)
        cas_payload = None if payload is None else payload.get("payload")
        derived_cid = (
            content_identity(cas_payload)
            if payload is not None
            else ""
        )
        if (
            isinstance(cas_payload, Mapping)
            and cas_payload.get("schema")
            == PLAN_REVISION_APPLY_RECEIPT_SCHEMA
        ):
            receipt_fields = {
                "schema",
                "receipt_cid",
                "intent_cid",
                "state",
                "revision_cid",
                "plan_root_cid",
                "delta_cid",
                "markdown_projection_cid",
                "duckdb_projection_cid",
                "prior_active_cid",
                "event_cursor",
                "expected_effects",
                "observed_effects",
                "deferred_item_keys",
                "activated_deferred_keys",
                "resumed",
                "quarantined",
                "committed",
                "reason_codes",
                "markdown_path",
                "duckdb_path",
            }
            if (
                set(cas_payload) != receipt_fields
                or cas_payload.get("receipt_cid") != parts[1]
                or not isinstance(cas_payload.get("resumed"), bool)
                or not isinstance(cas_payload.get("quarantined"), bool)
                or not isinstance(cas_payload.get("committed"), bool)
                or cas_payload.get("committed")
                != (cas_payload.get("state") in {"committed", "replayed"})
            ):
                raise ValueError("plan store receipt CAS projection is malformed")
            receipt_identity_body = {
                name: value
                for name, value in cas_payload.items()
                if name not in {"receipt_cid", "resumed", "committed"}
            }
            derived_cid = content_identity(receipt_identity_body)
        if (
            payload is None
            or set(payload) != {"schema", "cid", "media_type", "payload"}
            or payload.get("schema") != PLAN_REVISION_STORE_SCHEMA
            or payload.get("cid") != parts[1]
            or derived_cid != parts[1]
        ):
            raise ValueError("plan store CAS projection is malformed or mixed")
        return
    if len(parts) == 2 and parts[0] == "continuations":
        payload, _evidence = _read_stable_regular_json(artifact)
        if (
            payload is None
            or set(payload) != {
                "schema",
                "idempotency_key",
                "payload",
                "updated_at_ns",
                "continuation_cid",
            }
            or payload.get("schema") != PLAN_REVISION_CONTINUATION_SCHEMA
            or not isinstance(payload.get("idempotency_key"), str)
            or not isinstance(payload.get("payload"), Mapping)
            or isinstance(payload.get("updated_at_ns"), bool)
            or not isinstance(payload.get("updated_at_ns"), int)
            or parts[1]
            != _plan_bound_safe_store_filename(payload["idempotency_key"]) + ".json"
        ):
            raise ValueError("plan store continuation projection is malformed")
        body = dict(payload)
        continuation_cid = body.pop("continuation_cid")
        if content_identity(body) != continuation_cid:
            raise ValueError("plan store continuation content identity is mixed")
        return
    if len(parts) == 2 and parts[0] == "intents" and parts[1].endswith(".json"):
        payload, _evidence = _read_stable_regular_json(artifact)
        if payload is None or payload.get("schema") != PLAN_REVISION_INTENT_SCHEMA:
            raise ValueError("plan store intent projection is malformed")
        intent = PlanRevisionIntent.from_dict(payload)
        if intent.to_dict() != payload or parts[1] != f"{intent.intent_cid}.json":
            raise ValueError("plan store intent projection is mixed")
        return
    raise ValueError("plan store contains a noncanonical projection")


def _plan_bound_recovery_runtime_kind(
    artifact: Path,
    *,
    directory_projection: bool,
    runtime_roots: tuple[Path, Path, Path],
    owner_bound_artifacts: tuple[Path, ...],
    runtime_bindings: tuple[Mapping[str, Any], ...],
    state_dir: Path,
    state_prefix: str,
) -> str:
    """Classify only paths derived from the active manifest and handoffs."""

    state_root, worktree_root, merge_root = runtime_roots
    workspace_paths = {
        Path(str(binding.get("workspace_path") or ""))
        for binding in runtime_bindings
        if binding.get("workspace_path")
    }
    if artifact in owner_bound_artifacts:
        expected_workspace = artifact in workspace_paths
        if directory_projection != expected_workspace:
            return ""
        return "workspace" if expected_workspace else "file"
    if artifact.is_relative_to(worktree_root):
        relative = artifact.relative_to(worktree_root)
        if directory_projection and artifact in workspace_paths:
            return "workspace"
        entry_ids = {
            path.name.removeprefix("workspace-")
            for path in workspace_paths
            if path.name.startswith("workspace-")
        }
        if (
            not directory_projection
            and len(relative.parts) == 2
            and relative.parts[0] == ".pool-state"
            and (
                (
                    relative.stem in entry_ids
                    and relative.suffix in {".json", ".lock"}
                )
                or relative.name
                in {
                    f".{entry_id}.lock.update.lock"
                    for entry_id in entry_ids
                }
            )
        ):
            return "file"
        return ""
    if artifact.is_relative_to(merge_root):
        relative = artifact.relative_to(merge_root)
        request_ids = {
            str(binding.get("merge_request_id") or "")
            for binding in runtime_bindings
            if binding.get("merge_request_id")
        }
        dedupe_keys = {
            str(binding.get("merge_dedupe_key") or "")
            for binding in runtime_bindings
            if binding.get("merge_dedupe_key")
        }
        if directory_projection:
            return ""
        if relative.parts in {
            (".merge_queue.duckdb.lock",),
            ("merge_queue.duckdb",),
            ("train", "consumer.lock"),
        }:
            return "file"
        if (
            len(relative.parts) == 2
            and relative.parts[0] in {"pending", "processing", "completed", "failed"}
            and relative.stem in request_ids
            and relative.suffix == ".json"
        ):
            return "file"
        if (
            len(relative.parts) == 3
            and relative.parts[:2] == ("train", "receipts")
            and relative.stem in dedupe_keys
            and relative.suffix == ".json"
        ):
            return "file"
        return ""
    if not artifact.is_relative_to(state_root):
        return ""
    relative = artifact.relative_to(state_root)
    if relative.parts and relative.parts[0] == "plan-revision-store":
        if directory_projection:
            return ""
        _validate_plan_bound_store_projection(
            state_root / "plan-revision-store",
            artifact,
        )
        return "store"
    if directory_projection or len(relative.parts) < 2:
        return ""
    lane_name = relative.parts[0]
    lane_match = re.fullmatch(r"lane-([0-9]+)", lane_name)
    binding = None
    if lane_match is not None:
        lane_index = int(lane_match.group(1))
        binding = next(
            (
                item
                for item in runtime_bindings
                if item.get("lane_index") == lane_index
            ),
            None,
        )
    elif state_dir.parent == state_root and artifact.is_relative_to(state_dir):
        binding = next(
            (
                item
                for item in runtime_bindings
                if item.get("lane_id")
                and str(item["lane_id"]) == state_dir.name
            ),
            None,
        )
    if binding is None:
        return ""
    lane_index = int(binding["lane_index"])
    prefix_match = re.fullmatch(r"(.+)_lane_[0-9]+", state_prefix)
    lane_prefix = (
        f"{prefix_match.group(1)}_lane_{lane_index}"
        if prefix_match is not None
        else (state_prefix if artifact.is_relative_to(state_dir) else "")
    )
    if not lane_prefix:
        return ""
    lane_relative = PurePosixPath(*relative.parts[1:])
    name = lane_relative.name
    if len(lane_relative.parts) == 1 and name in {
        "task_queue.json",
        ".implementation.lock.update.lock",
        f".{lane_prefix}_events.jsonl.lock",
        f"{lane_prefix}_events.jsonl",
        f"{lane_prefix}_events.jsonl.manifest.json",
        f"{lane_prefix}_strategy.json",
        f"{lane_prefix}_task_state.json",
        f"{lane_prefix}_status.json",
    }:
        return "file"
    if len(lane_relative.parts) != 2 or lane_relative.parts[0] != "implementation_logs":
        return ""
    active_task_id = str(binding.get("active_task_id") or "")
    raw_attempt = binding.get("attempt")
    if (
        not active_task_id
        or active_task_id not in binding.get("task_ids", ())
        or isinstance(raw_attempt, bool)
        or not isinstance(raw_attempt, int)
        or raw_attempt < 1
    ):
        return ""
    safe_task = (
        re.sub(r"[^a-z0-9._-]+", "-", active_task_id.lower()).strip("-")
        or "task"
    )
    attempt = int(raw_attempt)
    if safe_task:
        if name in {
            f"{safe_task}-base-context-capsule.json",
            f"{safe_task}-base-context-receipt.json",
            f"{safe_task}-attempt-{attempt}-context-receipt.json",
            f"{safe_task}-attempt-{attempt}-provider-receipt.json",
            f"{safe_task}-attempt-{attempt}-task-execution-receipt.json",
            f"{safe_task}-attempt-{attempt}-retry-capsule.json",
            f"{safe_task}-attempt-{attempt}.log",
        }:
            return "file"
    return ""


def _snapshot_plan_bound_recovery_artifacts(
    *,
    root: Path,
    runtime_roots: tuple[Path, Path, Path],
    owner_bound_artifacts: tuple[Path, ...],
    runtime_bindings: tuple[Mapping[str, Any], ...],
    state_dir: Path,
    state_prefix: str,
) -> tuple[dict[str, Any], ...]:
    """Validate and bind every pre-existing non-store untracked artifact."""

    status = _plan_bound_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
        "--ignore-submodules=none",
    )
    if status.returncode != 0:
        raise ValueError("plan-bound recovery repository status is unavailable")
    evidence: list[dict[str, Any]] = []
    for raw_entry in str(status.stdout).split("\0"):
        if not raw_entry:
            continue
        if raw_entry[:3] not in {"?? ", "!! "}:
            raise ValueError("plan-bound recovery repository has tracked changes")
        relative_text = raw_entry[3:]
        directory_projection = relative_text.endswith("/")
        relative_text = relative_text[:-1] if directory_projection else relative_text
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != relative_text
        ):
            raise ValueError("plan-bound recovery has an unsafe untracked path")
        artifact = _lexical_contained_path(root, root / relative)
        kind = _plan_bound_recovery_runtime_kind(
            artifact,
            directory_projection=directory_projection,
            runtime_roots=runtime_roots,
            owner_bound_artifacts=owner_bound_artifacts,
            runtime_bindings=runtime_bindings,
            state_dir=state_dir,
            state_prefix=state_prefix,
        )
        if not kind:
            raise ValueError(
                "plan-bound recovery has a noncanonical runtime projection: "
                f"{relative_text!r}"
            )
        if kind == "store":
            continue
        evidence.append(
            _plan_bound_recovery_artifact_evidence(
                root,
                artifact,
                workspace=kind == "workspace",
            )
        )
    return tuple(sorted(evidence, key=lambda item: item["path"]))


def _validate_plan_bound_accepted_tree(
    *,
    accepted_tree_root: Path,
    source_head: str,
    source_tree: str,
    control_plane_pin: AgentImplementationControlPlanePin | None = None,
    recovery_repository_head: str = "",
    recovery_repository_tree: str = "",
    recovery_runtime_roots: tuple[Path, ...] = (),
    recovery_owner_bound_artifacts: tuple[Path, ...] = (),
    recovery_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> None:
    """Bind initial launches to HEAD and recovery to the sealed source object."""

    root = _canonical_accepted_tree_root(accepted_tree_root)
    if control_plane_pin is None:
        module_root = _canonical_accepted_tree_root(
            Path(__file__).absolute().parents[3]
        )
        if root != module_root:
            raise ValueError(
                "plan-bound accepted tree is not the live module root"
            )
    elif (
        control_plane_pin.source_head != source_head
        or control_plane_pin.source_tree != source_tree
    ):
        raise ValueError(
            "plan-bound accepted tree differs from the sealed control plane"
        )
    if re.fullmatch(r"[0-9a-f]{40,64}", source_head) is None:
        raise ValueError("plan-bound source HEAD is not a Git object identity")
    if re.fullmatch(r"[0-9a-f]{40,64}", source_tree) is None:
        raise ValueError("plan-bound source tree is not a Git object identity")
    if (
        not isinstance(recovery_repository_head, str)
        or not isinstance(recovery_repository_tree, str)
        or bool(recovery_repository_head) != bool(recovery_repository_tree)
    ):
        raise ValueError("plan-bound recovery repository identity is partial")
    if (
        not isinstance(recovery_runtime_roots, tuple)
        or any(not isinstance(path, Path) for path in recovery_runtime_roots)
        or not isinstance(recovery_owner_bound_artifacts, tuple)
        or any(
            not isinstance(path, Path)
            for path in recovery_owner_bound_artifacts
        )
        or not isinstance(recovery_artifacts, tuple)
        or any(not isinstance(item, Mapping) for item in recovery_artifacts)
    ):
        raise ValueError("plan-bound recovery runtime authority is malformed")
    source_object = _plan_bound_git(root, "rev-parse", f"{source_head}^{{tree}}")
    if (
        source_object.returncode != 0
        or str(source_object.stdout).strip() != source_tree
    ):
        raise ValueError("plan-bound sealed source object is unavailable or mixed")
    head = _plan_bound_git(root, "rev-parse", "HEAD")
    tree = _plan_bound_git(root, "rev-parse", "HEAD^{tree}")
    current_head = str(head.stdout).strip()
    current_tree = str(tree.stdout).strip()
    if recovery_repository_head:
        if control_plane_pin is None:
            raise ValueError(
                "plan-bound repository advance requires a sealed control plane"
            )
        if (
            re.fullmatch(r"[0-9a-f]{40}", recovery_repository_head) is None
            or re.fullmatch(r"[0-9a-f]{40}", recovery_repository_tree) is None
            or head.returncode != 0
            or tree.returncode != 0
            or current_head != recovery_repository_head
            or current_tree != recovery_repository_tree
        ):
            raise ValueError(
                "plan-bound recovery repository identity changed"
            )
        ancestor = _plan_bound_git(
            root,
            "merge-base",
            "--is-ancestor",
            source_head,
            recovery_repository_head,
        )
        status = _plan_bound_git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
            "--ignore-submodules=none",
        )
        if (
            ancestor.returncode != 0
            or status.returncode != 0
        ):
            raise ValueError(
                "plan-bound recovery repository is not a clean source descendant"
            )
        canonical_runtime_roots = tuple(
            _lexical_contained_path(root, path)
            for path in recovery_runtime_roots
        )
        if len(canonical_runtime_roots) != 3 or len(
            set(canonical_runtime_roots)
        ) != 3:
            raise ValueError(
                "plan-bound recovery runtime roots are absent or ambiguous"
            )
        canonical_owner_bound_artifacts = tuple(
            _lexical_contained_path(root, path)
            for path in recovery_owner_bound_artifacts
        )
        if (
            not canonical_owner_bound_artifacts
            or len(canonical_owner_bound_artifacts)
            != len(set(canonical_owner_bound_artifacts))
            or any(
                not any(
                    artifact.is_relative_to(runtime_root)
                    for runtime_root in canonical_runtime_roots
                )
                for artifact in canonical_owner_bound_artifacts
            )
        ):
            raise ValueError(
                "plan-bound recovery owner-bound artifact is absent or foreign"
            )

        expected_artifacts = {
            str(item.get("path") or ""): dict(item)
            for item in recovery_artifacts
        }
        if (
            not expected_artifacts
            or "" in expected_artifacts
            or len(expected_artifacts) != len(recovery_artifacts)
        ):
            raise ValueError(
                "plan-bound recovery artifact evidence is absent or ambiguous"
            )
        observed_artifacts: set[str] = set()
        for raw_entry in str(status.stdout).split("\0"):
            if not raw_entry:
                continue
            if raw_entry[:3] not in {"?? ", "!! "}:
                raise ValueError(
                    "plan-bound recovery repository has tracked changes"
                )
            relative_text = raw_entry[3:]
            directory_projection = relative_text.endswith("/")
            normalized_relative_text = (
                relative_text[:-1]
                if directory_projection
                else relative_text
            )
            relative = PurePosixPath(normalized_relative_text)
            if (
                not normalized_relative_text
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != normalized_relative_text
            ):
                raise ValueError(
                    "plan-bound recovery repository has an unsafe untracked path: "
                    f"{relative_text!r}"
                )
            artifact = _lexical_contained_path(root, root / relative)
            if not any(
                artifact.is_relative_to(runtime_root)
                for runtime_root in canonical_runtime_roots
            ):
                raise ValueError(
                    "plan-bound recovery repository has a foreign untracked path"
                )
            state_relative = (
                artifact.relative_to(canonical_runtime_roots[0])
                if artifact.is_relative_to(canonical_runtime_roots[0])
                else None
            )
            if (
                state_relative is not None
                and state_relative.parts[:1] == ("plan-revision-store",)
            ):
                if directory_projection:
                    raise ValueError("plan store directory projection is ambiguous")
                _validate_plan_bound_store_projection(
                    canonical_runtime_roots[0] / "plan-revision-store",
                    artifact,
                )
                continue
            evidence = expected_artifacts.get(normalized_relative_text)
            if evidence is None:
                if artifact not in canonical_owner_bound_artifacts:
                    raise ValueError(
                        "plan-bound recovery found an unauthenticated runtime "
                        f"artifact: {relative_text!r}"
                    )
                # These exact launch-owned paths may be created after the
                # immutable recovery decision (PID reservation/log only).
                # They are files, never workspace projections, and still
                # receive no executable or link exception.
                if directory_projection:
                    raise ValueError(
                        "plan-bound launch-owned artifact changed projection kind"
                    )
                _plan_bound_recovery_artifact_evidence(
                    root,
                    artifact,
                    workspace=False,
                )
                continue
            observed = _plan_bound_recovery_artifact_evidence(
                root,
                artifact,
                workspace=directory_projection,
            )
            if observed != evidence:
                raise ValueError(
                    "plan-bound recovery runtime artifact content identity changed"
                )
            observed_artifacts.add(normalized_relative_text)
        if observed_artifacts != set(expected_artifacts):
            raise ValueError("plan-bound recovery runtime artifact set changed")
        return
    if (
        head.returncode != 0
        or tree.returncode != 0
        or current_head != source_head
        or current_tree != source_tree
    ):
        raise ValueError("plan-bound accepted tree changed from the pinned source")
    for relative in (PLAN_BOUND_GATE_ENTRY_PATH, PLAN_BOUND_ACCEPTED_ENTRY_PATH):
        entry = _lexical_contained_path(
            root,
            root / relative,
            require_regular=True,
        )
        payload, _evidence = _read_stable_regular_bytes(entry, max_bytes=4_194_304)
        if payload is None:
            raise ValueError(f"plan-bound accepted entry is absent: {relative}")
        expected = _plan_bound_git(root, "rev-parse", f"{source_head}:{relative}")
        actual = _plan_bound_git(root, "hash-object", "--stdin", input_bytes=payload)
        actual_stdout = actual.stdout
        if isinstance(actual_stdout, bytes):
            actual_oid = actual_stdout.decode("ascii", errors="strict").strip()
        else:
            actual_oid = str(actual_stdout).strip()
        status = _plan_bound_git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative,
        )
        if (
            expected.returncode != 0
            or actual.returncode != 0
            or status.returncode != 0
            or actual_oid != str(expected.stdout).strip()
            or str(status.stdout).strip()
        ):
            raise ValueError(
                f"plan-bound accepted entry is not the clean pinned blob: {relative}"
            )


def _detached_docker_cleanup_binding(
    identity: ProcessIdentity,
    *,
    runner_pid: int,
    runner_start_ticks: int,
    runner_boot_id: str,
    records: Sequence[_DurableDockerCleanupBinding],
) -> tuple[str, str, str] | None:
    """Admit one exact reaper only through its parent-published record."""

    arguments = identity.argv
    if _DOCKER_CLEANUP_WATCHDOG_ARG not in arguments:
        return None
    if (
        arguments.count(_DOCKER_CLEANUP_WATCHDOG_ARG) != 1
        or arguments.count(_DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG) != 1
        or arguments.index(_DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG)
        >= arguments.index(_DOCKER_CLEANUP_WATCHDOG_ARG)
    ):
        raise ValueError("detached Docker cleanup command is ambiguous")

    def exact_option(name: str) -> str:
        if arguments.count(name) != 1:
            raise ValueError(f"detached Docker cleanup {name} is ambiguous")
        index = arguments.index(name)
        if index + 1 >= len(arguments):
            raise ValueError(f"detached Docker cleanup {name} is incomplete")
        return arguments[index + 1]

    provider = exact_option("--provider")
    docker_bin = exact_option("--docker-bin")
    container_name = exact_option("--container-name")
    raw_lease_root = exact_option("--lease-root")
    cidfile = exact_option("--cidfile")
    provider_home = exact_option("--provider-home")
    prompt_path = exact_option("--prompt-path")
    binding_path = Path(exact_option("--cleanup-binding-record"))
    if (
        provider not in {"codex", "grok"}
        or _DOCKER_CLEANUP_CONTAINER_RE.fullmatch(container_name) is None
        or not container_name.startswith(f"ipfs-accelerate-{provider}-")
    ):
        raise ValueError("detached Docker cleanup identity is invalid")
    try:
        docker_path = Path(docker_bin).resolve(strict=True)
        metadata = docker_path.stat()
    except OSError as exc:
        raise ValueError("detached Docker cleanup binary is unavailable") from exc
    if (
        docker_path not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
        or docker_path.name not in {"docker", "docker.exe"}
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or not os.access(docker_path, os.X_OK)
    ):
        raise ValueError("detached Docker cleanup binary is not trusted")
    lease_root = Path(raw_lease_root)
    if (
        not lease_root.is_absolute()
        or re.fullmatch(
            r"asref-(?:grok|codex)-container-[a-z0-9_]+",
            lease_root.name,
        )
        is None
    ):
        raise ValueError("detached Docker cleanup lease is not trusted")
    expected_record_path = (
        Path(identity.run_root)
        / _DOCKER_CLEANUP_BINDING_DIRECTORY
        / (hashlib.sha256(container_name.encode("ascii")).hexdigest() + ".json")
    )
    try:
        executable = Path(identity.executable).resolve(strict=True)
        executable_metadata = executable.stat()
        value = _read_durable_docker_cleanup_record(binding_path)
    except OSError as exc:
        raise ValueError("detached Docker cleanup authority is unavailable") from exc
    try:
        from .grok_cli_runner import _validated_docker_cleanup_root

        cleanup_root, cleanup_root_identity = _validated_docker_cleanup_root(
            lease_root=lease_root,
            provider_home=Path(provider_home),
            prompt_path=Path(prompt_path),
            expected_root=Path(str(value.get("cleanup_root") or "")),
            expected_identity=value.get("cleanup_root_identity"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("detached Docker cleanup root is invalid") from exc
    body = {key: item for key, item in value.items() if key != "record_id"}
    if (
        identity.pid <= 0
        or identity.start_time_ticks <= 0
        or identity.parent_pid != 1
        or executable != Path(sys.executable).resolve(strict=True)
        or not stat.S_ISREG(executable_metadata.st_mode)
        or executable_metadata.st_uid != 0
        or executable_metadata.st_mode & 0o022
        or binding_path != expected_record_path
        or value.get("schema") != _DOCKER_CLEANUP_BINDING_SCHEMA
        or value.get("record_id") != _docker_control_identity(body)
        or value.get("run_id") != identity.run_id
        or value.get("profile_id") != identity.profile_id
        or value.get("target_id") != identity.target_id
        or value.get("repository_root") != identity.repository_root
        or value.get("state_root") != identity.state_root
        or value.get("run_root") != identity.run_root
        or value.get("configuration_root") != identity.configuration_root
        or value.get("fencing_epoch") != identity.fencing_epoch
        or value.get("runner_pid") != runner_pid
        or value.get("runner_start_ticks") != runner_start_ticks
        or runner_boot_id != identity.boot_id
        or value.get("watchdog_pid") != identity.pid
        or value.get("watchdog_start_ticks") != identity.start_time_ticks
        or value.get("boot_id") != identity.boot_id
        or value.get("provider") != provider
        or value.get("docker_bin") != str(docker_path)
        or value.get("container_name") != container_name
        or value.get("cleanup_root") != str(cleanup_root)
        or value.get("cleanup_root_identity") != cleanup_root_identity
        or value.get("lease_root") != str(lease_root)
        or value.get("docker_config") != str(lease_root / "docker-config")
        or value.get("cidfile") != cidfile
        or value.get("provider_home") != provider_home
        or value.get("prompt_path") != prompt_path
        or value.get("binding_path") != str(binding_path)
    ):
        raise ValueError("detached Docker cleanup record is not authoritative")
    admitted = tuple(
        record
        for record in records
        if record.record_path == binding_path
        and record.runner_pid == runner_pid
        and record.runner_start_ticks == runner_start_ticks
        and record.watchdog_pid == identity.pid
        and record.watchdog_start_ticks == identity.start_time_ticks
        and record.boot_id == identity.boot_id
        and record.cleanup_root == cleanup_root
        and dict(record.cleanup_root_identity) == cleanup_root_identity
        and record.binding == (str(docker_path), container_name, str(lease_root))
    )
    if len(admitted) != 1:
        raise ValueError("detached Docker cleanup root lacks one admitted record")
    return str(docker_path), container_name, str(lease_root)


def _read_durable_docker_cleanup_record(
    path: Path,
    *,
    directory_anchor: _DurableCleanupDirectoryAnchor | None = None,
    directory_descriptor: int | None = None,
) -> dict[str, object]:
    """Read one canonical private lifecycle record without following links."""

    directory = path.parent
    directory_fd = -1
    try:
        if directory_anchor is not None and directory_descriptor is not None:
            raise ValueError("durable Docker cleanup directory authority repeats")
        if directory_descriptor is not None:
            if isinstance(directory_descriptor, bool) or directory_descriptor < 3:
                raise ValueError(
                    "durable Docker cleanup directory descriptor is invalid"
                )
            directory_fd = os.dup(directory_descriptor)
            opened_directory = os.fstat(directory_fd)
            named_directory = os.lstat(directory)
            if (
                opened_directory.st_dev,
                opened_directory.st_ino,
                stat.S_IFMT(opened_directory.st_mode),
                opened_directory.st_uid,
            ) != (
                named_directory.st_dev,
                named_directory.st_ino,
                stat.S_IFMT(named_directory.st_mode),
                named_directory.st_uid,
            ):
                os.close(directory_fd)
                raise ValueError(
                    "durable Docker cleanup directory identity changed"
                )
        elif directory_anchor is None:
            if directory.resolve(strict=True) != directory.absolute():
                raise ValueError("durable Docker cleanup directory is aliased")
            directory_fd = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        else:
            _validate_durable_cleanup_directory_anchor(
                directory_anchor,
                expected_path=directory,
            )
            directory_fd = os.dup(directory_anchor.descriptor)
        directory_metadata = os.fstat(directory_fd)
    except OSError as exc:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise ValueError("durable Docker cleanup directory is unavailable") from exc
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or directory_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
    ):
        os.close(directory_fd)
        raise ValueError("durable Docker cleanup directory is not private")
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        before = os.fstat(descriptor)
        try:
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size > _DOCKER_PRIVATE_CONTROL_MAX_BYTES
            ):
                raise ValueError("durable Docker cleanup record is unsafe")
            remaining = _DOCKER_PRIVATE_CONTROL_MAX_BYTES + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            final = os.stat(
                path.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError("durable Docker cleanup record is unreadable") from exc
    finally:
        os.close(directory_fd)
    snapshot = lambda item: (  # noqa: E731 - compact immutable stat projection.
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        len(raw) > _DOCKER_PRIVATE_CONTROL_MAX_BYTES
        or snapshot(before) != snapshot(after)
        or snapshot(after) != snapshot(final)
    ):
        raise ValueError("durable Docker cleanup record changed while read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        canonical = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("durable Docker cleanup record is invalid JSON") from exc
    if type(value) is not dict or raw != canonical:
        raise ValueError("durable Docker cleanup record is noncanonical")
    return value


def _stable_durable_docker_cleanup_entry_names(
    directory: Path,
    *,
    expected_metadata: os.stat_result,
    directory_anchor: _DurableCleanupDirectoryAnchor | None = None,
) -> tuple[str, ...]:
    """Return a stable namespace after bounded exact-writer publication.

    Private control records are published through a same-directory temporary
    name.  Seeing that exact writer state is contention, not a malformed
    authoritative record: wait for it to finish, but never admit state while
    it exists.  Every other unexpected name or unsafe temporary fails closed.
    """

    owns_directory_fd = directory_anchor is None
    directory_fd = -1
    try:
        if directory_anchor is None:
            directory_fd = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        else:
            _validate_durable_cleanup_directory_anchor(
                directory_anchor,
                expected_path=directory,
            )
            directory_fd = directory_anchor.descriptor
        opened = os.fstat(directory_fd)
        expected_identity = _cleanup_directory_stat_identity(expected_metadata)
        if _cleanup_directory_stat_identity(opened) != expected_identity:
            raise ValueError(
                "durable Docker cleanup directory identity changed"
            )
        deadline = (
            time.monotonic() + _DOCKER_CLEANUP_PUBLICATION_WAIT_SECONDS
        )
        while True:
            if directory_anchor is None:
                current = os.lstat(directory)
                if (
                    _cleanup_directory_stat_identity(os.fstat(directory_fd))
                    != expected_identity
                    or _cleanup_directory_stat_identity(current)
                    != expected_identity
                ):
                    raise ValueError(
                        "durable Docker cleanup directory identity changed"
                    )
            else:
                _validate_durable_cleanup_directory_anchor(
                    directory_anchor,
                    expected_path=directory,
                )
            entry_names = tuple(sorted(os.listdir(directory_fd)))
            publication_pending = False
            retry_enumeration = False
            for name in entry_names:
                if _DOCKER_CLEANUP_STABLE_ENTRY_RE.fullmatch(name) is not None:
                    continue
                temporary = _DOCKER_CLEANUP_ATOMIC_TEMP_RE.fullmatch(name)
                if temporary is None:
                    raise ValueError(
                        "durable Docker cleanup record set is invalid"
                    )
                try:
                    metadata = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    retry_enumeration = True
                    break
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size > _DOCKER_PRIVATE_CONTROL_MAX_BYTES
                    or metadata.st_nlink not in {1, 2}
                ):
                    raise ValueError(
                        "durable Docker cleanup atomic publication is unsafe"
                    )
                # Create-only publication briefly hard-links the completed
                # temporary inode to its final name before unlinking the
                # temporary.  Admit that as a wait state only when both names
                # are the exact same private regular file.
                if metadata.st_nlink == 2:
                    try:
                        published = os.stat(
                            temporary.group("target"),
                            dir_fd=directory_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        raise ValueError(
                            "durable Docker cleanup atomic publication is unsafe"
                        ) from None
                    if (
                        published.st_dev != metadata.st_dev
                        or published.st_ino != metadata.st_ino
                        or not stat.S_ISREG(published.st_mode)
                        or published.st_uid != os.geteuid()
                        or stat.S_IMODE(published.st_mode) != 0o600
                        or published.st_size
                        > _DOCKER_PRIVATE_CONTROL_MAX_BYTES
                    ):
                        raise ValueError(
                            "durable Docker cleanup atomic publication is unsafe"
                        )
                publication_pending = True
            if retry_enumeration:
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "durable Docker cleanup publication is contended"
                    )
                time.sleep(_DOCKER_CLEANUP_PUBLICATION_POLL_SECONDS)
                continue
            if not publication_pending:
                return entry_names
            if time.monotonic() >= deadline:
                raise ValueError(
                    "durable Docker cleanup publication is contended"
                )
            time.sleep(_DOCKER_CLEANUP_PUBLICATION_POLL_SECONDS)
    except OSError as exc:
        raise ValueError(
            "durable Docker cleanup directory cannot be enumerated"
        ) from exc
    finally:
        if owns_directory_fd and directory_fd >= 0:
            os.close(directory_fd)


def _durable_docker_cleanup_bindings(
    profile: LifecycleProfile,
    *,
    fencing_epoch: int | None,
    directory_anchor: _DurableCleanupDirectoryAnchor | None = None,
) -> tuple[_DurableDockerCleanupBinding, ...]:
    """Admit exact cleanup records that survive detached reaper death."""

    directory = Path(profile.run_root) / _DOCKER_CLEANUP_BINDING_DIRECTORY
    if directory_anchor is None:
        try:
            directory_metadata = os.lstat(directory)
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise ValueError("durable Docker cleanup directory is unavailable") from exc
    else:
        directory_metadata = _validate_durable_cleanup_directory_anchor(
            directory_anchor,
            expected_path=directory,
        )
    if (
        fencing_epoch is None
        or not stat.S_ISDIR(directory_metadata.st_mode)
        or directory_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        or directory.resolve(strict=True) != directory.absolute()
    ):
        raise ValueError("durable Docker cleanup directory is invalid")
    expected_fields = {
        "schema",
        "binding_state",
        "run_id",
        "profile_id",
        "target_id",
        "repository_root",
        "state_root",
        "run_root",
        "configuration_root",
        "fencing_epoch",
        "runner_pid",
        "runner_start_ticks",
        "watchdog_pid",
        "watchdog_start_ticks",
        "boot_id",
        "provider",
        "docker_bin",
        "docker_device",
        "docker_inode",
        "docker_mode",
        "docker_uid",
        "container_name",
        "cleanup_root",
        "cleanup_root_identity",
        "lease_root",
        "docker_config",
        "cidfile",
        "provider_home",
        "prompt_path",
        "effect_observation",
        "create_command_id",
        "create_cwd",
        "create_environment_id",
        "termination_fence",
        "path_identities",
        "binding_path",
        "record_id",
    }
    try:
        current_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        entry_names = _stable_durable_docker_cleanup_entry_names(
            directory,
            expected_metadata=directory_metadata,
            directory_anchor=directory_anchor,
        )
        entries = tuple(directory / name for name in entry_names)
    except OSError as exc:
        raise ValueError(
            "durable Docker cleanup directory cannot be enumerated"
        ) from exc
    if not current_boot_id or any(
        _DOCKER_CLEANUP_STABLE_ENTRY_RE.fullmatch(path.name) is None
        for path in entries
    ):
        raise ValueError("durable Docker cleanup record set is invalid")
    # Completion files contain a public hash but no mutation authority.  In
    # particular, never replay them before the matching active binding has
    # passed lifecycle, process-birth, CAS, and Docker-name checks below.  A
    # matching completion is consumed by _remove_durable_cleanup_record only
    # after those checks.  Standalone journals are inert and do not consume
    # active-record capacity; this prevents old journals from starving a
    # later exact binding while preserving fail-closed recovery semantics.
    entry_name_set = set(entry_names)
    dual_retirement_stems = {
        path.stem
        for path in entries
        if path.suffix == ".authority"
        and f"{path.stem}.json" in entry_name_set
    }
    # The exact .json/.authority hard-link pair is a crash state after the
    # terminal binding has already been retained.  It is validated below
    # under the canonical binding lock and never consumes active capacity.
    # A forged second name cannot evade admission: any non-identical pair or
    # pair without its exact completion/CAS evidence fails closed below.
    paths = tuple(
        path
        for path in entries
        if path.suffix == ".json"
        and path.stem not in dual_retirement_stems
    )
    if len(paths) > 128:
        raise ValueError("durable Docker cleanup record set is invalid")
    records: list[_DurableDockerCleanupBinding] = []
    binding_values: dict[str, Mapping[str, object]] = {}
    for path in paths:
        try:
            value = _read_durable_docker_cleanup_record(
                path,
                directory_anchor=directory_anchor,
            )
        except ValueError:
            # A successful watchdog atomically publishes its completion and
            # unlinks the active binding.  Enumeration can race that exact
            # transition.  Treat only a now-absent directory entry the same
            # as an unlink completed just before enumeration; an extant,
            # replaced, unreadable, or malformed entry remains fail-closed.
            try:
                if directory_anchor is None:
                    os.lstat(path)
                else:
                    os.stat(
                        path.name,
                        dir_fd=directory_anchor.descriptor,
                        follow_symlinks=False,
                    )
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(
                    "durable Docker cleanup record liveness is unknown"
                ) from exc
            raise
        try:
            if directory_anchor is None:
                record_metadata = os.lstat(path)
            else:
                record_metadata = os.stat(
                    path.name,
                    dir_fd=directory_anchor.descriptor,
                    follow_symlinks=False,
                )
        except FileNotFoundError:
            # The same admitted unlink may occur after a stable record read
            # but before its metadata projection.  No record is returned;
            # callers retain any earlier binding and independently require
            # exact Docker-name and lease-root absence before lane release.
            continue
        except OSError as exc:
            raise ValueError(
                "durable Docker cleanup record disappeared after read"
            ) from exc
        body = {key: item for key, item in value.items() if key != "record_id"}
        provider = str(value.get("provider") or "")
        container_name = str(value.get("container_name") or "")
        docker_bin = str(value.get("docker_bin") or "")
        lease_root = Path(str(value.get("lease_root") or ""))
        docker_config = Path(str(value.get("docker_config") or ""))
        cidfile = Path(str(value.get("cidfile") or ""))
        provider_home = Path(str(value.get("provider_home") or ""))
        prompt_path = Path(str(value.get("prompt_path") or ""))
        effect_observation = value.get("effect_observation")
        termination_fence = value.get("termination_fence")
        binding_state = str(value.get("binding_state") or "")
        command_bound = binding_state == "command_bound"
        create_cwd = (
            Path(str(value.get("create_cwd") or ""))
            if command_bound
            else Path()
        )
        path_identities = value.get("path_identities")
        path_identity_valid = bool(
            isinstance(path_identities, dict)
            and set(path_identities)
            == {
                "docker_config",
                "lease_root",
                "prompt_path",
                "provider_home",
            }
            and all(
                isinstance(item, dict)
                and set(item) == {"device", "inode", "mode", "uid"}
                and all(
                    type(item.get(name)) is int and int(item[name]) >= 0
                    for name in ("device", "inode", "mode", "uid")
                )
                and item.get("uid") == os.geteuid()
                for item in path_identities.values()
            )
            and all(
                stat.S_ISDIR(path_identities[name]["mode"])
                for name in ("docker_config", "lease_root", "provider_home")
            )
            and stat.S_ISREG(path_identities["prompt_path"]["mode"])
        )
        termination_fence_valid = False
        if isinstance(termination_fence, dict):
            try:
                from .grok_cli_runner import (
                    _validated_docker_termination_fence,
                )

                termination_fence_valid = bool(
                    not termination_fence
                    or _validated_docker_termination_fence(
                        termination_fence,
                        provider=provider,
                        container_name=container_name,
                    )
                    == termination_fence
                )
            except (KeyError, TypeError, ValueError):
                termination_fence_valid = False
        try:
            from .grok_cli_runner import _validated_docker_cleanup_root

            cleanup_root, cleanup_root_identity = (
                _validated_docker_cleanup_root(
                    lease_root=lease_root,
                    provider_home=provider_home,
                    prompt_path=prompt_path,
                    expected_root=Path(
                        str(value.get("cleanup_root") or "")
                    ),
                    expected_identity=value.get("cleanup_root_identity"),  # type: ignore[arg-type]
                )
            )
            cleanup_root_valid = bool(
                value.get("cleanup_root") == str(cleanup_root)
                and value.get("cleanup_root_identity")
                == cleanup_root_identity
            )
        except (TypeError, ValueError):
            cleanup_root = Path()
            cleanup_root_identity = {}
            cleanup_root_valid = False
        try:
            docker_path = Path(docker_bin).resolve(strict=True)
            docker_metadata = docker_path.stat()
            runner_pid = int(value.get("runner_pid"))
            runner_start_ticks = int(value.get("runner_start_ticks"))
            watchdog_pid = int(value.get("watchdog_pid"))
            watchdog_start_ticks = int(value.get("watchdog_start_ticks"))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("durable Docker cleanup identity is unavailable") from exc
        if (
            set(value) != expected_fields
            or value.get("schema") != _DOCKER_CLEANUP_BINDING_SCHEMA
            or binding_state not in {"prepared_no_dispatch", "command_bound"}
            or value.get("record_id") != _docker_control_identity(body)
            or value.get("run_id") != profile.run_id
            or value.get("profile_id") != profile.profile_id
            or value.get("target_id") != profile.target_id
            or value.get("repository_root") != profile.repository_root
            or value.get("state_root") != profile.state_root
            or value.get("run_root") != profile.run_root
            or value.get("configuration_root") != profile.configuration_root
            or value.get("fencing_epoch") != fencing_epoch
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                str(value.get("boot_id") or ""),
            )
            is None
            or runner_pid <= 0
            or runner_start_ticks <= 0
            or watchdog_pid <= 0
            or watchdog_start_ticks <= 0
            or provider not in {"codex", "grok"}
            or (effect_observation and provider != "codex")
            or docker_path
            not in {Path("/usr/bin/docker"), Path("/usr/local/bin/docker")}
            or value.get("docker_device") != docker_metadata.st_dev
            or value.get("docker_inode") != docker_metadata.st_ino
            or value.get("docker_mode") != docker_metadata.st_mode
            or value.get("docker_uid") != docker_metadata.st_uid
            or docker_metadata.st_uid != 0
            or docker_metadata.st_mode & 0o022
            or _DOCKER_CLEANUP_CONTAINER_RE.fullmatch(container_name) is None
            or not container_name.startswith(f"ipfs-accelerate-{provider}-")
            or path.name
            != hashlib.sha256(container_name.encode("ascii")).hexdigest()
            + ".json"
            or value.get("binding_path") != str(path)
            or not cleanup_root_valid
            or not lease_root.name.startswith(f"asref-{provider}-container-")
            or docker_config != lease_root / "docker-config"
            or cidfile != lease_root / "container.cid"
            or not provider_home.name.startswith(f"asref-{provider}-home-")
            or not prompt_path.name.startswith("asref-grok-prompt-")
            or not isinstance(effect_observation, dict)
            or set(effect_observation)
            not in (
                set(),
                {
                    "logical_attempt_id",
                    "provider_attempt_store",
                    "provider_attempt_store_identity",
                },
            )
            or any(
                not isinstance(item, str) or not item
                for item in effect_observation.values()
            )
            or not path_identity_valid
            or not termination_fence_valid
            or (
                command_bound
                and (
                    re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(value.get("create_command_id") or ""),
                    )
                    is None
                    or not create_cwd.is_absolute()
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(value.get("create_environment_id") or ""),
                    )
                    is None
                )
            )
            or (
                not command_bound
                and (
                    value.get("create_command_id") != ""
                    or value.get("create_cwd") != ""
                    or value.get("create_environment_id") != ""
                )
            )
        ):
            raise ValueError("durable Docker cleanup binding drifted")
        records.append(
            _DurableDockerCleanupBinding(
                docker_bin=str(docker_path),
                provider=provider,
                container_name=container_name,
                cleanup_root=cleanup_root,
                cleanup_root_identity=MappingProxyType(
                    dict(cleanup_root_identity)
                ),
                lease_root=lease_root,
                docker_config=docker_config,
                cidfile=cidfile,
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation=MappingProxyType(dict(effect_observation)),
                create_command_id=str(value["create_command_id"]),
                create_cwd=create_cwd,
                create_environment_id=str(value["create_environment_id"]),
                termination_fence=MappingProxyType(
                    dict(termination_fence)
                ),
                binding_state=binding_state,
                path_identities=MappingProxyType(
                    {
                        name: MappingProxyType(dict(identity))
                        for name, identity in path_identities.items()
                    }
                ),
                runner_pid=runner_pid,
                runner_start_ticks=runner_start_ticks,
                watchdog_pid=watchdog_pid,
                watchdog_start_ticks=watchdog_start_ticks,
                boot_id=str(value["boot_id"]),
                record_path=path,
                record_device=record_metadata.st_dev,
                record_inode=record_metadata.st_ino,
                record_id=str(value["record_id"]),
            )
        )
        binding_values[path.stem] = value
    authority_paths = tuple(
        path for path in entries if path.suffix == ".authority"
    )
    for authority_path in authority_paths:
        stem = authority_path.stem
        binding_path = directory / f"{stem}.json"
        completion_path = directory / f"{stem}.complete"
        dual_retirement = stem in dual_retirement_stems
        if completion_path.name not in entry_name_set:
            raise ValueError(
                "retired Docker cleanup authority is not exclusive"
            )
        completion = _read_durable_docker_cleanup_record(
            completion_path,
            directory_anchor=directory_anchor,
        )
        binding_identity = completion.get("binding_identity")
        cleanup_intent = completion.get("cleanup_intent")
        resources = completion.get("resources")
        if dual_retirement:
            binding_record = completion.get("binding_record")
            if not isinstance(binding_record, Mapping) or not isinstance(
                binding_identity,
                Mapping,
            ):
                raise ValueError(
                    "retiring Docker cleanup authority is malformed"
                )
            from .grok_cli_runner import (
                _cleanup_binding_retirement_pair_matches,
                _docker_binding_lock_descriptor,
            )

            retirement_lock = _docker_binding_lock_descriptor(binding_path)
            try:
                pair_matches = _cleanup_binding_retirement_pair_matches(
                    retirement_lock.directory_fd,
                    binding_name=binding_path.name,
                    authority_name=authority_path.name,
                    binding_identity=binding_identity,  # type: ignore[arg-type]
                    binding_record=binding_record,
                )
                retirement_lock.assert_current()
            finally:
                retirement_lock.close()
            if not pair_matches:
                raise ValueError(
                    "retiring Docker cleanup authority pair drifted"
                )
            authority_record = dict(binding_record)
        else:
            authority_record = _read_durable_docker_cleanup_record(
                authority_path,
                directory_anchor=directory_anchor,
            )
        try:
            authority_metadata = (
                os.lstat(authority_path)
                if directory_anchor is None
                else os.stat(
                    authority_path.name,
                    dir_fd=directory_anchor.descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError as exc:
            raise ValueError(
                "retired Docker cleanup authority is unavailable"
            ) from exc
        if (
            not isinstance(binding_identity, Mapping)
            or not isinstance(cleanup_intent, Mapping)
            or not isinstance(resources, list)
            or authority_record.get("binding_path") != str(binding_path)
            or completion.get("binding_record") != authority_record
            or not _owned_cleanup_path_matches(
                authority_metadata,
                directory=False,
                identity=binding_identity,
            )
        ):
            raise ValueError("retired Docker cleanup authority drifted")
        from .grok_cli_runner import (
            _cleanup_completion_value,
            _cleanup_intent_terminal_authority,
            _cleanup_path_quarantine,
            _cleanup_progress_matches,
        )

        terminal_authority = _cleanup_intent_terminal_authority(
            cleanup_intent
        )
        if _cleanup_completion_value(
            binding_path=binding_path,
            binding_identity=binding_identity,
            binding_record=authority_record,
            terminal_cleanup_authority=terminal_authority,
        ) != completion:
            raise ValueError("retired Docker cleanup completion drifted")
        for resource in resources:
            if (
                not isinstance(resource, Mapping)
                or not isinstance(resource.get("identity"), Mapping)
                or not isinstance(resource.get("directory"), bool)
            ):
                raise ValueError(
                    "retired Docker cleanup resource is malformed"
                )
            resource_path = Path(str(resource.get("path") or ""))
            quarantine, owned, marker, _tombstone = _cleanup_path_quarantine(
                resource_path,
                directory=bool(resource["directory"]),
                identity=resource["identity"],  # type: ignore[arg-type]
            )
            if any(
                os.path.lexists(candidate)
                for candidate in (
                    resource_path,
                    owned,
                    marker,
                    quarantine,
                )
            ):
                raise ValueError(
                    "retired Docker cleanup resource remains materialized"
                )
        if terminal_authority is not None:
            observation = authority_record.get("effect_observation")
            if not isinstance(observation, Mapping):
                raise ValueError(
                    "retired Docker cleanup CAS locator is malformed"
                )
            try:
                from ..control.provider_attempt_store import (
                    DurableProviderAttemptCAS,
                )

                attempt_store = DurableProviderAttemptCAS(
                    str(observation["provider_attempt_store"]),
                    expected_directory_identity=str(
                        observation["provider_attempt_store_identity"]
                    ),
                    create_if_missing=False,
                )
                terminal = attempt_store.observe(
                    str(observation["logical_attempt_id"])
                )
            except (KeyError, OSError, ValueError) as exc:
                raise ValueError(
                    "retired Docker cleanup terminal CAS is unavailable"
                ) from exc
            if (
                terminal is None
                or terminal.state != "terminal"
                or terminal.terminal_cleanup_authority
                != terminal_authority
                or not _cleanup_progress_matches(
                    terminal.terminal_cleanup_progress,
                    intent=cleanup_intent,
                    completion_id=str(completion.get("completion_id") or ""),
                )
            ):
                raise ValueError(
                    "retired Docker cleanup terminal CAS drifted"
                )
        if dual_retirement:
            from .grok_cli_runner import (
                _docker_binding_lock_descriptor,
                _retire_cleanup_binding_authority,
            )

            retirement_lock = _docker_binding_lock_descriptor(binding_path)
            try:
                retired = _retire_cleanup_binding_authority(
                    binding_path,
                    binding_identity=binding_identity,  # type: ignore[arg-type]
                    binding_record=authority_record,
                    binding_lock=retirement_lock,
                )
            finally:
                retirement_lock.close()
            if not retired:
                raise ValueError(
                    "retiring Docker cleanup authority did not converge"
                )
    for sidecar in entries:
        match = re.fullmatch(
            r"([0-9a-f]{64})\.(lock|remove-dispatched)",
            sidecar.name,
        )
        if match is None:
            continue
        stem, kind = match.groups()
        if not (
            f"{stem}.json" in entry_name_set
            or f"{stem}.authority" in entry_name_set
            or f"{stem}.complete" in entry_name_set
        ):
            raise ValueError("durable Docker cleanup sidecar is orphaned")
        try:
            metadata = (
                os.lstat(sidecar)
                if directory_anchor is None
                else os.stat(
                    sidecar.name,
                    dir_fd=directory_anchor.descriptor,
                    follow_symlinks=False,
                )
            )
        except OSError as exc:
            raise ValueError(
                "durable Docker cleanup sidecar is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (kind == "lock" and metadata.st_size != 0)
        ):
            raise ValueError("durable Docker cleanup sidecar is unsafe")
        if kind == "lock":
            continue
        dispatched = _read_durable_docker_cleanup_record(
            sidecar,
            directory_anchor=directory_anchor,
        )
        source_binding = binding_values.get(stem)
        if source_binding is None:
            completion_path = directory / f"{stem}.complete"
            completion = _read_durable_docker_cleanup_record(
                completion_path,
                directory_anchor=directory_anchor,
            )
            source_binding = completion.get("binding_record")
            binding_identity = completion.get("binding_identity")
            if (
                not isinstance(source_binding, Mapping)
                or not isinstance(binding_identity, Mapping)
            ):
                raise ValueError(
                    "durable Docker removal completion is malformed"
                )
            cleanup_intent = completion.get("cleanup_intent")
            if not isinstance(cleanup_intent, Mapping):
                raise ValueError(
                    "durable Docker removal completion intent is malformed"
                )
            from .grok_cli_runner import (
                _cleanup_completion_value,
                _cleanup_intent_terminal_authority,
            )

            if _cleanup_completion_value(
                binding_path=directory / f"{stem}.json",
                binding_identity=binding_identity,
                binding_record=source_binding,
                terminal_cleanup_authority=(
                    _cleanup_intent_terminal_authority(cleanup_intent)
                ),
            ) != completion:
                raise ValueError(
                    "durable Docker removal completion drifted"
                )
        termination_fence = source_binding.get("termination_fence")
        if not isinstance(termination_fence, Mapping) or not termination_fence:
            raise ValueError("durable Docker removal fence is absent")
        from .grok_cli_runner import _validated_docker_removal_dispatch

        _validated_docker_removal_dispatch(
            dispatched,
            binding_path=directory / f"{stem}.json",
            binding_record=source_binding,
            termination_fence=termination_fence,
        )
    if len({record.binding for record in records}) != len(records):
        raise ValueError("durable Docker cleanup bindings repeat a container")
    if directory_anchor is not None:
        _validate_durable_cleanup_directory_anchor(
            directory_anchor,
            expected_path=directory,
        )
    return tuple(records)


def _validated_durable_docker_create_journal(
    record: _DurableDockerCleanupBinding,
    *,
    directory_descriptor: int | None = None,
) -> dict[str, object]:
    """Return one fully canonical command-bound Docker-create journal."""

    if record.binding_state != "command_bound":
        raise ValueError("durable Docker create journal is not command-bound")
    journal_path = record.lease_root / _DOCKER_CREATE_JOURNAL_NAME
    try:
        value = _read_durable_docker_cleanup_record(
            journal_path,
            directory_descriptor=directory_descriptor,
        )
    except FileNotFoundError as exc:
        raise ValueError("durable Docker create journal is absent") from exc
    expected_fields = {
        "schema",
        "provider",
        "docker_bin",
        "docker_config",
        "container_name",
        "cidfile",
        "cwd",
        "environment_id",
        "image_id",
        "argv",
        "command_id",
        "state",
        "issuer_process_birth",
        "returncode",
        "stdout_hex",
        "stderr_hex",
        "journal_id",
    }
    body = {key: item for key, item in value.items() if key != "journal_id"}
    command_body = {
        key: value[key]
        for key in (
            "provider",
            "docker_bin",
            "docker_config",
            "container_name",
            "cidfile",
            "cwd",
            "environment_id",
            "image_id",
            "argv",
        )
    }
    argv = value.get("argv")
    exact_argv = False
    if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
        try:
            from .grok_cli_runner import _docker_create_command_identity

            canonical_command_id, canonical_command_body = (
                _docker_create_command_identity(
                    provider=record.provider,
                    docker_bin=record.docker_bin,
                    docker_config=record.docker_config,
                    container_name=record.container_name,
                    cidfile=record.cidfile,
                    cwd=record.create_cwd,
                    environment_id=record.create_environment_id,
                    expected_image=str(value.get("image_id") or ""),
                    argv=argv,
                )
            )
            exact_argv = bool(
                canonical_command_id == record.create_command_id
                and canonical_command_body == command_body
            )
        except (OSError, ValueError):
            exact_argv = False
    try:
        stdout = bytes.fromhex(str(value.get("stdout_hex") or ""))
        stderr = bytes.fromhex(str(value.get("stderr_hex") or ""))
    except ValueError as exc:
        raise ValueError("durable Docker create output is invalid") from exc
    state = str(value.get("state") or "")
    returncode = value.get("returncode")
    issuer = value.get("issuer_process_birth")
    issuer_required = state in {
        "create_inflight",
        "create_observed",
        "create_failed_observed",
        "create_outcome_unknown",
    }
    valid_issuer = bool(
        isinstance(issuer, dict)
        and set(issuer) == {"pid", "start_time_ticks", "boot_id", "parent_pid"}
        and type(issuer.get("pid")) is int
        and issuer["pid"] > 0
        and type(issuer.get("start_time_ticks")) is int
        and issuer["start_time_ticks"] > 0
        and type(issuer.get("parent_pid")) is int
        and issuer["parent_pid"] == record.watchdog_pid
        and isinstance(issuer.get("boot_id"), str)
        and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(issuer.get("boot_id") or ""),
        )
        is not None
    )
    if (
        set(value) != expected_fields
        or value.get("schema") != _DOCKER_CREATE_JOURNAL_SCHEMA
        or value.get("provider") != record.provider
        or value.get("docker_bin") != record.docker_bin
        or value.get("docker_config") != str(record.docker_config)
        or value.get("container_name") != record.container_name
        or value.get("cidfile") != str(record.cidfile)
        or value.get("cwd") != str(record.create_cwd)
        or value.get("environment_id") != record.create_environment_id
        or value.get("command_id") != record.create_command_id
        or not exact_argv
        or _docker_control_identity(command_body) != record.create_command_id
        or value.get("journal_id") != _docker_control_identity(body)
        or state
        not in {
            "prepared",
            "create_armed",
            "create_inflight",
            "create_observed",
            "create_failed_observed",
            "create_outcome_unknown",
            "prepared_abandoned",
        }
        or issuer_required != valid_issuer
        or (
            not issuer_required
            and issuer != {}
        )
        or len(stdout) > _DOCKER_CLEANUP_INSPECTION_MAX_BYTES
        or len(stderr) > _DOCKER_CLEANUP_INSPECTION_MAX_BYTES
        or (
            state == "create_observed"
            and (type(returncode) is not int or returncode != 0)
        )
        or (
            state == "create_failed_observed"
            and (type(returncode) is not int or returncode == 0)
        )
        or (
            state == "create_outcome_unknown"
            and (type(returncode) is not int or returncode == 0)
        )
        or (
            state
            not in {
                "create_observed",
                "create_failed_observed",
                "create_outcome_unknown",
            }
            and (returncode is not None or stdout or stderr)
        )
    ):
        raise ValueError("durable Docker create journal drifted")
    return value


def _durable_docker_create_state(
    record: _DurableDockerCleanupBinding,
) -> str:
    if record.binding_state == "prepared_no_dispatch":
        # The watchdog published this authority before readiness and the
        # runner never atomically upgraded it to a command-bound dispatch.
        return "prepared_no_dispatch"
    value = _validated_durable_docker_create_journal(record)
    return str(value["state"])


def _durable_docker_create_issuer_gone(
    record: _DurableDockerCleanupBinding,
) -> bool | None:
    """Prove the sole durable Docker-create issuer birth is no longer live."""

    state = _durable_docker_create_state(record)
    if state not in {"create_inflight", "create_outcome_unknown"}:
        return None
    value = _read_durable_docker_cleanup_record(
        record.lease_root / _DOCKER_CREATE_JOURNAL_NAME
    )
    issuer = value.get("issuer_process_birth")
    if not isinstance(issuer, dict):
        return None
    birth = WorktreeProcessBirthIdentity.from_dict(issuer)
    liveness = owner_liveness(birth)
    if liveness is OwnerLiveness.UNKNOWN:
        return None
    return liveness is OwnerLiveness.DEAD


def _exact_process_birth_alive(
    *,
    pid: int,
    start_ticks: int,
    boot_id: str,
) -> bool | None:
    """Return exact liveness, exact death, or an unobservable fail-closed state."""

    liveness = owner_liveness(
        WorktreeProcessBirthIdentity(
            pid=pid,
            start_time_ticks=start_ticks,
            boot_id=boot_id,
        )
    )
    if liveness is OwnerLiveness.UNKNOWN:
        return None
    return liveness is OwnerLiveness.ALIVE


def _durable_cleanup_watchdog_alive(
    record: _DurableDockerCleanupBinding,
) -> bool | None:
    """Observe the receipt-bound watchdog birth independently of profiles."""

    return _exact_process_birth_alive(
        pid=record.watchdog_pid,
        start_ticks=record.watchdog_start_ticks,
        boot_id=record.boot_id,
    )


def _durable_cleanup_runner_alive(
    record: _DurableDockerCleanupBinding,
) -> bool | None:
    """Observe the record-bound runner birth, not only its numeric PID."""

    return _exact_process_birth_alive(
        pid=record.runner_pid,
        start_ticks=record.runner_start_ticks,
        boot_id=record.boot_id,
    )


def _private_marker(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        raise ValueError("Docker cleanup marker is unavailable")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError("Docker cleanup marker is invalid")
    return True


def _durable_cleanup_cas_allows_reap(
    record: _DurableDockerCleanupBinding,
) -> bool:
    local_owned = _private_marker(record.lease_root / "cas-owned")
    local_terminal = _private_marker(record.lease_root / "cas-terminal")
    observation = dict(record.effect_observation)
    if not observation:
        return not local_owned and not local_terminal
    try:
        from ..control.provider_attempt_store import DurableProviderAttemptCAS

        store = DurableProviderAttemptCAS(
            observation["provider_attempt_store"],
            expected_directory_identity=(
                observation["provider_attempt_store_identity"]
            ),
            create_if_missing=False,
        )
        reservation = store.observe(observation["logical_attempt_id"])
    except (KeyError, OSError, ValueError):
        return False
    if reservation is None or reservation.state == "reserved":
        return not local_owned and not local_terminal
    if reservation.state not in {"effect_started", "quarantined", "terminal"}:
        return False
    launch = reservation.effect_launch_receipt
    cleanup = launch.get("cleanup_receipt")
    if not isinstance(cleanup, Mapping):
        return False
    try:
        from .grok_cli_runner import _recorded_codex_cleanup_identity

        observed_root, observed_config, observed_name = (
            _recorded_codex_cleanup_identity(launch)
        )
    except (ImportError, OSError, ValueError):
        return False
    local_receipt = bool(
        observed_root == record.lease_root
        and observed_config == record.docker_config
        and observed_name == record.container_name
        and launch.get("container_name") == record.container_name
        and cleanup.get("lease_root") == str(record.lease_root)
        and cleanup.get("docker_config") == str(record.docker_config)
        and cleanup.get("watchdog_pid") == record.watchdog_pid
        and cleanup.get("watchdog_start_ticks") == record.watchdog_start_ticks
    )
    if local_receipt:
        # The durable CAS terminal receipt is authoritative.  The local
        # marker is only a watchdog wake-up hint and a crash may occur after
        # terminal CAS publication but before that redundant marker write.
        return reservation.state == "terminal"
    # A foreign winner proves this exact local lease lost before Docker start,
    # but a local ownership marker would contradict that proof.
    return not local_owned and not local_terminal


def _exact_docker_name_absent(
    record: _DurableDockerCleanupBinding,
    *,
    deadline: float,
) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    temporary_config: Path | None = None
    config_descriptor = -1
    config_still_bound = True
    try:
        if os.path.lexists(record.docker_config):
            config_descriptor = os.open(
                record.docker_config,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            if not _owned_cleanup_path_matches(
                os.fstat(config_descriptor),
                directory=True,
                identity=record.path_identities["docker_config"],
            ):
                return False
            inspection_config = Path(f"/proc/self/fd/{config_descriptor}")
            inspection_pass_fds = (config_descriptor,)
        else:
            # The exact Docker-config directory is nested under the lease and
            # may already be tombstoned.  Docker's local Unix socket does not
            # require provider credentials, so use a fresh private config for
            # the independent exact-name absence observation rather than
            # recreating or trusting a removed resource path.
            temporary_config = Path(
                tempfile.mkdtemp(prefix="aseh-docker-recovery-config-")
            )
            temporary_config.chmod(0o700)
            inspection_config = temporary_config
            inspection_pass_fds = ()
        observed = subprocess.run(
            [
                record.docker_bin,
                f"--host={_DOCKER_LOCAL_HOST}",
                "--config",
                str(inspection_config),
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"name=^/{record.container_name}$",
                "--format",
                "{{.Names}}",
            ],
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=min(2.0, max(0.05, remaining)),
            check=False,
            pass_fds=inspection_pass_fds,
        )
        if config_descriptor >= 0:
            config_still_bound = _owned_cleanup_path_matches(
                os.lstat(record.docker_config),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
    except (KeyError, OSError, subprocess.TimeoutExpired):
        return False
    finally:
        if config_descriptor >= 0:
            os.close(config_descriptor)
        if temporary_config is not None:
            shutil.rmtree(temporary_config, ignore_errors=True)
    return bool(
        observed.returncode == 0
        and len(observed.stdout) <= _DOCKER_CLEANUP_INSPECTION_MAX_BYTES
        and not observed.stdout.strip()
        and config_still_bound
    )


def _exact_docker_container_materialized(
    record: _DurableDockerCleanupBinding,
    *,
    deadline: float,
) -> Mapping[str, object] | None:
    """Attest one exact Docker effect and its current Linux process scope."""

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    lease_descriptor = -1
    config_descriptor = -1
    cid_descriptor = -1
    try:
        if (
            record.binding_state != "command_bound"
            or record.docker_config != record.lease_root / "docker-config"
            or record.cidfile != record.lease_root / "container.cid"
        ):
            return None
        lease_descriptor = os.open(
            record.lease_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not _owned_cleanup_path_matches(
            os.fstat(lease_descriptor),
            directory=True,
            identity=record.path_identities["lease_root"],
        ):
            return None
        config_descriptor = os.open(
            "docker-config",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=lease_descriptor,
        )
        if not _owned_cleanup_path_matches(
            os.fstat(config_descriptor),
            directory=True,
            identity=record.path_identities["docker_config"],
        ):
            return None
        # Validate the journal again from the already-bound lease descriptor.
        # The earlier state check schedules this branch; it is never authority
        # for the image or command used by the Docker inspection below.
        journal = _validated_durable_docker_create_journal(
            record,
            directory_descriptor=lease_descriptor,
        )
        image_id = str(journal.get("image_id") or "")
        if (
            journal.get("state") != "create_observed"
            or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        ):
            return None
        cid_descriptor = os.open(
            "container.cid",
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=lease_descriptor,
        )
        cid_before = os.fstat(cid_descriptor)
        if (
            not stat.S_ISREG(cid_before.st_mode)
            or cid_before.st_uid != os.geteuid()
            or cid_before.st_nlink != 1
            or cid_before.st_size > 128
        ):
            return None
        raw_cid = os.read(cid_descriptor, 129)
        cid_after = os.fstat(cid_descriptor)
        cid_named = os.stat(
            "container.cid",
            dir_fd=lease_descriptor,
            follow_symlinks=False,
        )
        cid_snapshot = lambda item: (  # noqa: E731 - immutable stat projection.
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if (
            len(raw_cid) > 128
            or cid_snapshot(cid_before) != cid_snapshot(cid_after)
            or cid_snapshot(cid_after) != cid_snapshot(cid_named)
            or re.fullmatch(
                r"[0-9a-f]{64}", raw_cid.decode("ascii").strip()
            )
            is None
            or not _owned_cleanup_path_matches(
                os.lstat(record.lease_root),
                directory=True,
                identity=record.path_identities["lease_root"],
            )
            or not _owned_cleanup_path_matches(
                os.lstat(record.docker_config),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
        ):
            return None
        container_id = raw_cid.decode("ascii").strip()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        from .grok_cli_runner import _attest_exact_docker_execution

        termination_fence = _attest_exact_docker_execution(
            docker_bin=record.docker_bin,
            docker_config=Path(f"/proc/self/fd/{config_descriptor}"),
            provider=record.provider,
            container_name=record.container_name,
            container_id=container_id,
            image_id=image_id,
            timeout=min(2.0, max(0.05, remaining)),
            pass_fds=(config_descriptor,),
        )
        cid_after_attestation = os.fstat(cid_descriptor)
        cid_named_after_attestation = os.stat(
            "container.cid",
            dir_fd=lease_descriptor,
            follow_symlinks=False,
        )
        if (
            cid_snapshot(cid_after)
            != cid_snapshot(cid_after_attestation)
            or cid_snapshot(cid_after_attestation)
            != cid_snapshot(cid_named_after_attestation)
            or not _owned_cleanup_path_matches(
                os.fstat(lease_descriptor),
                directory=True,
                identity=record.path_identities["lease_root"],
            )
            or not _owned_cleanup_path_matches(
                os.fstat(config_descriptor),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
            or not _owned_cleanup_path_matches(
                os.lstat(record.lease_root),
                directory=True,
                identity=record.path_identities["lease_root"],
            )
            or not _owned_cleanup_path_matches(
                os.lstat(record.docker_config),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
        ):
            return None
        return MappingProxyType(dict(termination_fence))
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        subprocess.TimeoutExpired,
        UnicodeError,
        ValueError,
    ):
        return None
    finally:
        for descriptor in (
            cid_descriptor,
            config_descriptor,
            lease_descriptor,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _publish_durable_docker_termination_fence(
    record: _DurableDockerCleanupBinding,
    termination_fence: Mapping[str, object],
) -> _DurableDockerCleanupBinding:
    """Win the one CAS transition that authorizes a single Docker rm."""

    from .grok_cli_runner import (
        _cleanup_path_identity,
        _publish_docker_termination_fence_binding,
        _validated_docker_termination_fence,
    )

    if record.termination_fence:
        raise ValueError("Docker termination fence is already published")
    admitted_fence = _validated_docker_termination_fence(
        termination_fence,
        provider=record.provider,
        container_name=record.container_name,
    )
    expected_identity = _cleanup_path_identity(
        record.record_path,
        directory=False,
    )
    if (
        expected_identity.get("device") != record.record_device
        or expected_identity.get("inode") != record.record_inode
    ):
        raise ValueError("Docker cleanup binding lost its CAS identity")
    admitted, refreshed_identity = _publish_docker_termination_fence_binding(
        record_path=record.record_path,
        expected_record_id=record.record_id,
        expected_identity=expected_identity,
        provider=record.provider,
        docker_bin=record.docker_bin,
        docker_config=record.docker_config,
        container_name=record.container_name,
        cidfile=record.cidfile,
        lease_root=record.lease_root,
        provider_home=record.provider_home,
        prompt_path=record.prompt_path,
        effect_observation=record.effect_observation,
        runner_pid=record.runner_pid,
        runner_start_ticks=record.runner_start_ticks,
        watchdog_pid=record.watchdog_pid,
        watchdog_start_ticks=record.watchdog_start_ticks,
        create_command_id=record.create_command_id,
        create_cwd=record.create_cwd,
        create_environment_id=record.create_environment_id,
        termination_fence=admitted_fence,
    )
    refreshed_record_id = str(admitted.get("record_id") or "")
    if (
        admitted.get("termination_fence") != admitted_fence
        or refreshed_record_id == record.record_id
        or re.fullmatch(r"sha256:[0-9a-f]{64}", refreshed_record_id) is None
        or set(refreshed_identity) != {"device", "inode", "mode", "uid"}
        or any(
            type(refreshed_identity.get(name)) is not int
            for name in ("device", "inode", "mode", "uid")
        )
    ):
        raise ValueError("Docker termination fence CAS was not refreshed")
    return replace(
        record,
        termination_fence=MappingProxyType(dict(admitted_fence)),
        record_device=int(refreshed_identity["device"]),
        record_inode=int(refreshed_identity["inode"]),
        record_id=refreshed_record_id,
    )


def _remove_fenced_durable_docker_effect(
    record: _DurableDockerCleanupBinding,
    *,
    deadline: float,
    issue_removal: bool,
) -> bool:
    """Issue the sole CID-bound rm using the record-bound Docker config FD."""

    if not record.termination_fence:
        return False
    config_descriptor = -1
    try:
        config_descriptor = os.open(
            record.docker_config,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not _owned_cleanup_path_matches(
            os.fstat(config_descriptor),
            directory=True,
            identity=record.path_identities["docker_config"],
        ):
            return False
        from .grok_cli_runner import _remove_exact_docker_container

        _remove_exact_docker_container(
            docker_bin=record.docker_bin,
            docker_config=Path(f"/proc/self/fd/{config_descriptor}"),
            container_name=record.container_name,
            settle_for_creation=False,
            deadline=deadline,
            termination_fence=record.termination_fence,
            pass_fds=(config_descriptor,),
            issue_removal=issue_removal,
        )
        return bool(
            _owned_cleanup_path_matches(
                os.fstat(config_descriptor),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
            and _owned_cleanup_path_matches(
                os.lstat(record.docker_config),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
        )
    except (KeyError, OSError, ValueError):
        return False
    finally:
        if config_descriptor >= 0:
            os.close(config_descriptor)


def _arm_fenced_durable_docker_removal(
    record: _DurableDockerCleanupBinding,
) -> bool:
    """Use the canonical per-binding marker to admit at most one Docker rm."""

    from .grok_cli_runner import (
        _arm_docker_removal_once,
        _cleanup_path_identity,
        _read_private_control_record,
    )

    raw = _read_private_control_record(
        record.record_path.parent,
        record.record_path.name,
    )
    if (
        raw is None
        or raw.get("record_id") != record.record_id
        or raw.get("termination_fence") != dict(record.termination_fence)
    ):
        raise ValueError("Docker removal binding is unavailable")
    identity = _cleanup_path_identity(record.record_path, directory=False)
    if (
        identity.get("device") != record.record_device
        or identity.get("inode") != record.record_inode
    ):
        raise ValueError("Docker removal binding identity drifted")
    return _arm_docker_removal_once(
        binding_path=record.record_path,
        expected_binding_identity=identity,
        binding_record=raw,
        termination_fence=record.termination_fence,
    )


def _fenced_durable_docker_effect_absent(
    record: _DurableDockerCleanupBinding,
    *,
    deadline: float,
) -> bool:
    """Reconcile a prior fenced rm without ever dispatching it again."""

    from .grok_cli_runner import (
        _docker_termination_scope_quiescent,
        _validated_docker_termination_fence,
    )

    try:
        fence = _validated_docker_termination_fence(
            record.termination_fence,
            provider=record.provider,
            container_name=record.container_name,
        )
    except (KeyError, TypeError, ValueError):
        return False
    temporary_config: Path | None = None
    config_descriptor = -1
    bound_record_config = False
    try:
        if os.path.lexists(record.docker_config):
            config_descriptor = os.open(
                record.docker_config,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            if not _owned_cleanup_path_matches(
                os.fstat(config_descriptor),
                directory=True,
                identity=record.path_identities["docker_config"],
            ):
                return False
            bound_record_config = True
        else:
            temporary_config = Path(
                tempfile.mkdtemp(prefix="aseh-docker-fence-recovery-config-")
            )
            temporary_config.chmod(0o700)
            config_descriptor = os.open(
                temporary_config,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        inspection_config = Path(f"/proc/self/fd/{config_descriptor}")
        absence_samples = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            observed_id = subprocess.run(
                [
                    record.docker_bin,
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "--config",
                    str(inspection_config),
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"id={fence['container_id']}",
                    "--format",
                    "{{.ID}}",
                ],
                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=min(2.0, max(0.05, remaining)),
                check=False,
                pass_fds=(config_descriptor,),
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            observed_name = subprocess.run(
                [
                    record.docker_bin,
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "--config",
                    str(inspection_config),
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{record.container_name}$",
                    "--format",
                    "{{.Names}}",
                ],
                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=min(2.0, max(0.05, remaining)),
                check=False,
                pass_fds=(config_descriptor,),
            )
            config_still_bound = bool(
                not bound_record_config
                or (
                    _owned_cleanup_path_matches(
                        os.fstat(config_descriptor),
                        directory=True,
                        identity=record.path_identities["docker_config"],
                    )
                    and _owned_cleanup_path_matches(
                        os.lstat(record.docker_config),
                        directory=True,
                        identity=record.path_identities["docker_config"],
                    )
                )
            )
            exact_absence = bool(
                observed_id.returncode == 0
                and observed_name.returncode == 0
                and len(observed_id.stdout)
                <= _DOCKER_CLEANUP_INSPECTION_MAX_BYTES
                and len(observed_name.stdout)
                <= _DOCKER_CLEANUP_INSPECTION_MAX_BYTES
                and not observed_id.stdout.strip()
                and not observed_name.stdout.strip()
                and config_still_bound
                and _docker_termination_scope_quiescent(fence)
            )
            absence_samples = absence_samples + 1 if exact_absence else 0
            if absence_samples >= 2:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))
    except (KeyError, OSError, subprocess.TimeoutExpired, ValueError):
        return False
    finally:
        if config_descriptor >= 0:
            os.close(config_descriptor)
        if temporary_config is not None:
            shutil.rmtree(temporary_config, ignore_errors=True)


def _unmaterialized_docker_name_absent(
    record: _DurableDockerCleanupBinding,
    *,
    deadline: float,
) -> bool:
    """Prove name absence for a state that carries no admitted Docker effect."""

    if not os.path.lexists(record.docker_config):
        return _exact_docker_name_absent(record, deadline=deadline)
    config_descriptor = -1
    try:
        config_descriptor = os.open(
            record.docker_config,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not _owned_cleanup_path_matches(
            os.fstat(config_descriptor),
            directory=True,
            identity=record.path_identities["docker_config"],
        ):
            return False
        from .grok_cli_runner import _remove_exact_docker_container

        # With no termination fence the canonical helper never dispatches rm;
        # it only requires two stable exact-name absence observations.
        _remove_exact_docker_container(
            docker_bin=record.docker_bin,
            docker_config=Path(f"/proc/self/fd/{config_descriptor}"),
            container_name=record.container_name,
            settle_for_creation=False,
            deadline=deadline,
            pass_fds=(config_descriptor,),
        )
        return bool(
            _owned_cleanup_path_matches(
                os.fstat(config_descriptor),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
            and _owned_cleanup_path_matches(
                os.lstat(record.docker_config),
                directory=True,
                identity=record.path_identities["docker_config"],
            )
        )
    except (KeyError, OSError, ValueError):
        return False
    finally:
        if config_descriptor >= 0:
            os.close(config_descriptor)


def _remove_durable_cleanup_record(
    record: _DurableDockerCleanupBinding,
) -> bool:
    """Publish and converge the canonical post-Docker cleanup transition."""

    try:
        current = _read_durable_docker_cleanup_record(record.record_path)
        if current.get("record_id") != record.record_id:
            return False
        from .grok_cli_runner import (
            _cleanup_path_identity,
            _finalize_verified_cleanup_completion,
        )

        binding_identity = _cleanup_path_identity(
            record.record_path,
            directory=False,
        )
        if (
            binding_identity.get("device") != record.record_device
            or binding_identity.get("inode") != record.record_inode
        ):
            return False
        cleanup_store: object | None = None
        terminal_reservation: object | None = None
        observation = dict(record.effect_observation)
        if observation:
            from ..control.provider_attempt_store import (
                DurableProviderAttemptCAS,
            )

            cleanup_store = DurableProviderAttemptCAS(
                str(observation["provider_attempt_store"]),
                expected_directory_identity=str(
                    observation["provider_attempt_store_identity"]
                ),
                create_if_missing=False,
            )
            terminal_reservation = cleanup_store.observe(
                str(observation["logical_attempt_id"])
            )
            if (
                terminal_reservation is None
                or terminal_reservation.state != "terminal"
                or terminal_reservation.terminal_cleanup_authority.get(
                    "binding_record_id"
                )
                != record.record_id
            ):
                return False
        return _finalize_verified_cleanup_completion(
            binding_path=record.record_path,
            binding_identity=binding_identity,
            binding_record=current,
            terminal_cleanup_store=cleanup_store,
            terminal_cleanup_reservation=terminal_reservation,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False


def _owned_cleanup_path_matches(
    metadata: os.stat_result,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> bool:
    from .grok_cli_runner import (
        _owned_cleanup_path_matches as canonical_path_matches,
    )

    return canonical_path_matches(
        metadata,
        directory=directory,
        identity=identity,
    )


def _remove_owned_cleanup_path(
    path: Path,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> bool:
    """Use the runner's one canonical replayable inode-removal protocol."""

    from .grok_cli_runner import (
        _remove_or_admit_cleanup_tombstone as canonical_remove_path,
    )

    return canonical_remove_path(
        path,
        directory=directory,
        identity=identity,
    )


def _discard_owned_cleanup_tombstone(
    path: Path,
    *,
    directory: bool,
    identity: Mapping[str, int],
) -> bool:
    from .grok_cli_runner import (
        _discard_owned_cleanup_tombstone as canonical_discard_tombstone,
    )

    return canonical_discard_tombstone(
        path,
        directory=directory,
        identity=identity,
    )


def _reconcile_durable_docker_cleanup(
    record: _DurableDockerCleanupBinding,
    *,
    deadline: float,
) -> bool:
    """Recover one dead reaper through the immutable effect/CAS boundary."""

    try:
        if not _durable_cleanup_cas_allows_reap(record):
            return False
        # Recheck both exact births inside the mutation boundary.  Outer
        # shutdown scans use the same observations for scheduling, but must
        # not be the sole authority for replaying a durable record.
        if (
            _durable_cleanup_runner_alive(record) is not False
            or _durable_cleanup_watchdog_alive(record) is not False
        ):
            return False
        try:
            current_boot_id = Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            return False
        if not current_boot_id:
            return False
        same_boot = current_boot_id == record.boot_id
        already_fenced = bool(record.termination_fence)
        # A canonical termination fence was published only after an independently
        # attested successful create.  Once present, it remains the authority even
        # if the same-UID-writable public journal is missing or rewritten.
        create_state = (
            "create_observed"
            if already_fenced
            else _durable_docker_create_state(record)
        )
        allowed_states = {
            "prepared_no_dispatch",
            "prepared",
            "prepared_abandoned",
            "create_armed",
            "create_inflight",
            "create_observed",
            "create_failed_observed",
            "create_outcome_unknown",
        }
        if create_state not in allowed_states:
            return False
        if already_fenced and (
            record.binding_state != "command_bound"
            or create_state != "create_observed"
        ):
            return False
        if (
            same_boot
            and record.binding_state == "command_bound"
            and create_state != "create_observed"
        ):
            # A same-boot public journal cannot downgrade an armed, failed, or
            # unknown create into authority to release the reserved name.
            return False
        if create_state in {"create_inflight", "create_outcome_unknown"} and (
            same_boot
            or _durable_docker_create_issuer_gone(record) is not True
        ):
            return False
        lease_present = record.lease_root.exists()
        if lease_present:
            if not _owned_cleanup_path_matches(
                os.lstat(record.lease_root),
                directory=True,
                identity=record.path_identities["lease_root"],
            ) or not _owned_cleanup_path_matches(
                os.lstat(record.docker_config),
                directory=True,
                identity=record.path_identities["docker_config"],
            ):
                return False
        working_record = record
        if create_state == "create_observed":
            if already_fenced:
                # Fence publication happens while the provider is live, before
                # cleanup.  The separate durable dispatch marker determines
                # whether this recovery owns the sole rm or must observe only.
                issue_removal = _arm_fenced_durable_docker_removal(
                    working_record
                )
                effect_absent = _remove_fenced_durable_docker_effect(
                    working_record,
                    deadline=deadline,
                    issue_removal=issue_removal,
                )
            else:
                # The canonical CAS publisher validates the current boot.  An
                # unfenced effect from an earlier boot cannot acquire deletion
                # authority from a mutable name-only observation.
                if not same_boot or not lease_present:
                    return False
                termination_fence = _exact_docker_container_materialized(
                    working_record,
                    deadline=deadline,
                )
                if termination_fence is None:
                    return False
                working_record = _publish_durable_docker_termination_fence(
                    working_record,
                    termination_fence,
                )
                issue_removal = _arm_fenced_durable_docker_removal(
                    working_record
                )
                effect_absent = _remove_fenced_durable_docker_effect(
                    working_record,
                    deadline=deadline,
                    issue_removal=issue_removal,
                )
        else:
            if already_fenced:
                return False
            # These states carry no admitted completed effect.  The canonical
            # helper receives no fence and therefore performs observation only.
            effect_absent = _unmaterialized_docker_name_absent(
                working_record,
                deadline=deadline,
            )
        if not effect_absent:
            return False
        # The canonical finalizer owns the stable per-binding lock across all
        # resource tombstones and binding retirement.  Fence publication may
        # replace the record inode, so finalize only against the refreshed CAS
        # identity, never the stale input one.
        return _remove_durable_cleanup_record(working_record)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def _detached_docker_cleanup_bindings(
    tree: ProcessTreeSnapshot,
    *,
    process_pid: int,
    process_start_ticks: int,
    process_boot_id: str,
    records: Sequence[_DurableDockerCleanupBinding] = (),
) -> tuple[tuple[str, str, str], ...]:
    """Admit every auxiliary profile root as one exact cleanup reaper."""

    bindings: list[tuple[str, str, str]] = []
    for root in tree.roots:
        if root.pid == process_pid:
            continue
        binding = _detached_docker_cleanup_binding(
            root,
            runner_pid=process_pid,
            runner_start_ticks=process_start_ticks,
            runner_boot_id=process_boot_id,
            records=records,
        )
        if binding is None:
            raise ValueError("managed lifecycle profile has an unknown auxiliary root")
        bindings.append(binding)
    if len(bindings) != len(set(bindings)):
        raise ValueError("managed lifecycle profile repeats a cleanup binding")
    return tuple(bindings)


def _detached_docker_cleanup_absence_verified(
    bindings: Sequence[tuple[str, str, str]],
    *,
    deadline: float,
) -> bool:
    """Independently prove every receipt-bound Docker name is absent."""

    for docker_bin, container_name, lease_root in bindings:
        if Path(lease_root).exists():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            observed = subprocess.run(
                [
                    docker_bin,
                    f"--host={_DOCKER_LOCAL_HOST}",
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{container_name}$",
                    "--format",
                    "{{.Names}}",
                ],
                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=min(2.0, max(0.05, remaining)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if (
            observed.returncode != 0
            or len(observed.stdout) > _DOCKER_CLEANUP_INSPECTION_MAX_BYTES
            or observed.stdout.strip()
        ):
            return False
    return True


def _release_process_cleanup_directory_anchor(
    process: subprocess.Popen[bytes],
) -> bool:
    """Close a parent-held cleanup namespace only after a verified fence."""

    anchor = getattr(
        process,
        "_agent_supervisor_cleanup_directory_anchor",
        None,
    )
    if anchor is None:
        return not bool(
            getattr(
                process,
                "_agent_supervisor_cleanup_directory_anchor_required",
                False,
            )
        )
    if not isinstance(anchor, _DurableCleanupDirectoryAnchor):
        return False
    try:
        os.close(anchor.descriptor)
    except OSError:
        return False
    try:
        delattr(process, "_agent_supervisor_cleanup_directory_anchor")
    except AttributeError:
        return False
    process._agent_supervisor_cleanup_directory_anchor_required = False
    return True


def _terminate_managed_process(
    process: subprocess.Popen[bytes] | None,
    *,
    grace_seconds: float,
) -> tuple[bool, tuple[int, ...]]:
    """Fence the exact marker-bound tree associated with ``process``."""

    if process is None:
        return True, ()
    profile = getattr(process, "_agent_supervisor_lifecycle_profile", None)
    if not isinstance(profile, LifecycleProfile):
        # A caller-created Popen has no durable run/profile binding.  Refuse to
        # turn its PID into signal authority.
        return False, ()
    adapter = LinuxProcessAdapter()
    tree = adapter.snapshot(profile)
    stored_identity = getattr(
        process,
        "_agent_supervisor_process_identity",
        None,
    )
    daemon_kernel_fence = getattr(
        process,
        "_agent_supervisor_managed_daemon_kernel_fence",
        None,
    )
    cleanup_directory_anchor = getattr(
        process,
        "_agent_supervisor_cleanup_directory_anchor",
        None,
    )
    cleanup_directory_anchor_required = bool(
        getattr(
            process,
            "_agent_supervisor_cleanup_directory_anchor_required",
            False,
        )
    )
    if cleanup_directory_anchor_required and not isinstance(
        cleanup_directory_anchor,
        _DurableCleanupDirectoryAnchor,
    ):
        return False, ()
    if isinstance(cleanup_directory_anchor, _DurableCleanupDirectoryAnchor):
        try:
            _validate_durable_cleanup_directory_anchor(
                cleanup_directory_anchor,
                expected_path=(
                    Path(profile.run_root)
                    / _DOCKER_CLEANUP_BINDING_DIRECTORY
                ),
            )
        except ValueError:
            return False, ()
    plan_bound_profile = "--plan-bound-dispatch" in profile.argv
    if plan_bound_profile and not isinstance(
        cleanup_directory_anchor,
        _DurableCleanupDirectoryAnchor,
    ):
        # A restarted/adopted owner cannot reconstruct this pre-effect kernel
        # namespace identity from a public path.  Until a Quack-bound persisted
        # directory generation is available, missing live FD authority is
        # UNKNOWN and cannot release a plan-bound lane.
        return False, ()
    if plan_bound_profile and not isinstance(
        daemon_kernel_fence,
        _ManagedDaemonKernelFence,
    ):
        # Older/injected Popen handles may lack the in-memory convenience
        # projection.  Reconstruct it only from the immutable lifecycle
        # profile plus the exact kernel birth identity; the public PID
        # sidecar is never promoted into signal authority.
        if not isinstance(stored_identity, ProcessIdentity):
            return False, ()
        try:
            daemon_kernel_fence = _managed_daemon_kernel_fence_from_profile(
                profile,
                stored_identity,
            )
        except (OSError, TypeError, ValueError):
            return False, ()
    process_member = next(
        (item for item in tree.members if item.pid == process.pid), None
    )

    # Credential-bearing supervisors deliberately become non-dumpable after
    # launch.  Linux then denies the lifecycle adapter's /proc/<pid>/environ
    # scan even to the same-UID parent, so an empty/partial profile snapshot is
    # not proof that the direct Popen child exited.  Reattach only the exact
    # birth identity captured by start_track; a bare numeric PID is never
    # promoted into signal authority.
    omitted_live_root = process.poll() is None and process_member is None
    if omitted_live_root:
        if not isinstance(stored_identity, ProcessIdentity):
            return False, tuple(item.pid for item in tree.members)
        if (
            stored_identity.pid != process.pid
            or stored_identity.profile_id != profile.profile_id
            or stored_identity.run_id != profile.run_id
        ):
            raise ProcessIdentityMismatch(
                "managed Popen birth identity does not match its lifecycle profile"
            )
        if not adapter.identity_alive(stored_identity):
            # poll() still reports an unreaped direct child, but the immutable
            # PID/start-time identity cannot be observed.  This is UNKNOWN,
            # never a successful fence.
            return False, tuple(item.pid for item in tree.members)
        tree = replace(
            tree,
            members=(*tree.members, stored_identity),
            tree_id="",
        )
        process_member = stored_identity

    root_ids = {item.pid for item in tree.roots}
    if process_member is not None and process.pid not in root_ids:
        raise ProcessIdentityMismatch(
            "managed Popen does not identify the marker-bound tree root"
        )
    member_pids = tuple(item.pid for item in tree.members)
    cleanup_bindings_valid = True
    cleanup_bindings: tuple[tuple[str, str, str], ...] = ()
    admitted_cleanup_roots: dict[
        tuple[int, int, str],
        tuple[str, tuple[str, str, str]],
    ] = {}
    fencing_values = {
        item.fencing_epoch
        for item in (
            *tree.members,
            *((stored_identity,) if isinstance(stored_identity, ProcessIdentity) else ()),
        )
    }
    fencing_epoch = next(iter(fencing_values)) if len(fencing_values) == 1 else None
    runner_identity = (
        stored_identity
        if isinstance(stored_identity, ProcessIdentity)
        and stored_identity.pid == process.pid
        else process_member
    )

    def split_provider_and_cleanup(
        snapshot: ProcessTreeSnapshot,
    ) -> tuple[
        tuple[tuple[str, str, str], ...],
        tuple[ProcessIdentity, ...],
    ]:
        if not isinstance(runner_identity, ProcessIdentity):
            raise ValueError("managed runner birth identity is unavailable")
        admitted_records = _durable_docker_cleanup_bindings(
            profile,
            fencing_epoch=fencing_epoch,
            directory_anchor=(
                cleanup_directory_anchor
                if isinstance(
                    cleanup_directory_anchor,
                    _DurableCleanupDirectoryAnchor,
                )
                else None
            ),
        )
        auxiliary_roots = tuple(
            root for root in snapshot.roots if root.pid != process.pid
        )
        try:
            bindings = _detached_docker_cleanup_bindings(
                snapshot,
                process_pid=process.pid,
                process_start_ticks=runner_identity.start_time_ticks,
                process_boot_id=runner_identity.boot_id,
                records=admitted_records,
            )
        except ValueError:
            retained: list[tuple[str, str, str]] = []
            for root in auxiliary_roots:
                prior = admitted_cleanup_roots.get(
                    (root.pid, root.start_time_ticks, root.boot_id)
                )
                if prior is None or prior[0] != root.identity_id:
                    raise
                retained.append(prior[1])
            bindings = tuple(retained)
        else:
            for root, binding in zip(auxiliary_roots, bindings, strict=True):
                admitted_cleanup_roots[
                    (root.pid, root.start_time_ticks, root.boot_id)
                ] = (root.identity_id, binding)
        cleanup_root_pids = {
            root.pid for root in snapshot.roots if root.pid != process.pid
        }
        by_pid = {item.pid: item for item in snapshot.members}

        def belongs_to_cleanup_root(item: ProcessIdentity) -> bool:
            cursor = item
            seen: set[int] = set()
            while cursor.pid not in seen:
                if cursor.pid in cleanup_root_pids:
                    return True
                seen.add(cursor.pid)
                parent = by_pid.get(cursor.parent_pid)
                if parent is None:
                    return False
                cursor = parent
            return True

        return bindings, tuple(
            item
            for item in snapshot.members
            if not belongs_to_cleanup_root(item)
        )

    try:
        cleanup_bindings, primary_members = split_provider_and_cleanup(tree)
    except ValueError:
        # Continue fencing every already-proven primary member, but never
        # promote an unknown auxiliary root into successful lane shutdown.
        cleanup_bindings_valid = False
        primary_members = tree.members

    primary_tree = replace(tree, members=primary_members, tree_id="")
    # Freeze every provider-launching member before the first durable record
    # scan.  Detached cleanup workers remain runnable so an already armed
    # create can reach a definitive result and reap itself.
    frozen: dict[tuple[int, int], ProcessIdentity] = {}
    signal_exact = getattr(adapter, "_signal_exact", None)
    try:
        if callable(signal_exact) and not omitted_live_root:
            # A child can fork between the first snapshot and its SIGSTOP.
            # Freeze every newly observed provider member and require two
            # identical snapshots before inspecting durable cleanup records.
            # Cleanup workers remain live, including their sole Docker-create
            # child, so an armed request can still reach a definite result.
            stable_snapshot_ids: frozenset[tuple[int, int]] | None = None
            stable_samples = 0
            for _sample in range(8):
                live_bindings, live_primary = split_provider_and_cleanup(tree)
                cleanup_bindings = tuple(
                    sorted({*cleanup_bindings, *live_bindings})
                )
                for identity in sorted(
                    live_primary,
                    key=lambda item: item.pid,
                    reverse=True,
                ):
                    birth = (identity.pid, identity.start_time_ticks)
                    if birth not in frozen and adapter.identity_alive(identity):
                        signal_exact(identity, signal.SIGSTOP)
                        frozen[birth] = identity
                observed = adapter.snapshot(profile)
                observed_bindings, observed_primary = split_provider_and_cleanup(
                    observed
                )
                cleanup_bindings = tuple(
                    sorted({*cleanup_bindings, *observed_bindings})
                )
                observed_ids = frozenset(
                    (item.pid, item.start_time_ticks)
                    for item in observed_primary
                )
                if observed_ids == stable_snapshot_ids:
                    stable_samples += 1
                else:
                    stable_snapshot_ids = observed_ids
                    stable_samples = 1
                tree = observed
                primary_members = observed_primary
                primary_tree = replace(
                    observed,
                    members=tuple(frozen.values()),
                    tree_id="",
                )
                if stable_samples >= 2:
                    break
            else:
                cleanup_bindings_valid = False
            member_pids = tuple(
                sorted({*member_pids, *(item.pid for item in tree.members)})
            )
        durable_records = _durable_docker_cleanup_bindings(
            profile,
            fencing_epoch=fencing_epoch,
            directory_anchor=(
                cleanup_directory_anchor
                if isinstance(
                    cleanup_directory_anchor,
                    _DurableCleanupDirectoryAnchor,
                )
                else None
            ),
        )
        if callable(signal_exact) and not omitted_live_root:
            # Queue cooperative termination while every provider-launching
            # member is still stopped.  SIGCONT cannot open a post-scan create
            # race because SIGTERM is already pending for the exact birth.
            for identity in sorted(
                frozen.values(),
                key=lambda item: item.pid,
                reverse=True,
            ):
                if adapter.identity_alive(identity):
                    signal_exact(identity, signal.SIGTERM)
    except (OSError, ProcessIdentityMismatch, ValueError):
        durable_records = ()
        cleanup_bindings_valid = False
    finally:
        for identity in reversed(tuple(frozen.values())):
            try:
                if adapter.identity_alive(identity):
                    signal_exact(identity, signal.SIGCONT)
            except (OSError, ProcessIdentityMismatch):
                cleanup_bindings_valid = False

    cleanup_binding_set = {
        *cleanup_bindings,
        *(record.binding for record in durable_records),
    }

    if not tree.members and not durable_records:
        if isinstance(daemon_kernel_fence, _ManagedDaemonKernelFence):
            if not isinstance(stored_identity, ProcessIdentity):
                return False, ()
            if not _fence_managed_daemon_from_kernel_binding(
                daemon_kernel_fence,
                root_identity=stored_identity,
                grace_seconds=grace_seconds,
            ):
                return False, ()
            strict_state, _strict_tree = (
                _strict_plan_bound_process_fence_observation(
                    profile,
                    stored_identity,
                )
            )
            fenced = bool(
                process.poll() is not None
                and cleanup_bindings_valid
                and strict_state == "dead"
            )
            if fenced and not _release_process_cleanup_directory_anchor(process):
                fenced = False
            return fenced, ()
        fenced = process.poll() is not None and cleanup_bindings_valid
        if fenced and not _release_process_cleanup_directory_anchor(process):
            fenced = False
        return fenced, ()

    if omitted_live_root:
        # An opaque root cannot safely receive a cooperative TERM before the
        # complete tree is fenced.  Its handler can exit while leaving a
        # detached, independently grouped daemon behind; after reparenting,
        # neither a marker scan nor a parent walk can rediscover that daemon.
        # Freeze and repeatedly discover the exact PID/start-time tree first,
        # then kill every captured group/member before capacity is released.
        if not terminate_pid_tree(
            stored_identity.pid,
            grace_seconds=max(0.0, grace_seconds),
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=stored_identity.process_group_id,
            expected_root_start_time_ticks=(
                stored_identity.start_time_ticks
            ),
        ):
            return False, member_pids
        try:
            process.wait(timeout=max(1.0, grace_seconds))
        except (ChildProcessError, OSError, subprocess.TimeoutExpired):
            pass

    # Finish any exact members captured before the cooperative root signal.
    # This is also the ordinary visible-tree path.
    if primary_tree.members:
        adapter.terminate(
            primary_tree,
            grace_seconds=grace_seconds,
            deadline_ms=max(
                1,
                int(max(0.0, grace_seconds) * 1000) + 1_000,
            ),
        )
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except (ChildProcessError, OSError, subprocess.TimeoutExpired):
        pass

    if isinstance(daemon_kernel_fence, _ManagedDaemonKernelFence):
        if not isinstance(stored_identity, ProcessIdentity):
            return False, member_pids
        if not _fence_managed_daemon_from_kernel_binding(
            daemon_kernel_fence,
            root_identity=stored_identity,
            grace_seconds=grace_seconds,
        ):
            return False, member_pids

    # Unknown Docker-create recovery requires an eight-second post-issuer
    # settle window plus bounded control-plane calls.  Keep that reserve
    # inside this caller-owned deadline; ordinary clean shutdown exits early.
    deadline = time.monotonic() + max(12.0, max(0.1, grace_seconds) + 1.0)
    stable_empty_samples = 0
    while time.monotonic() < deadline:
        current_fencing_epoch = fencing_epoch
        exact_root_alive = bool(
            isinstance(stored_identity, ProcessIdentity)
            and stored_identity.pid == process.pid
            and adapter.identity_alive(stored_identity)
        )
        current_tree = adapter.snapshot(profile)
        try:
            current_live_bindings, current_primary_members = (
                split_provider_and_cleanup(current_tree)
            )
            current_fencing_values = {
                item.fencing_epoch for item in current_tree.members
            }
            current_fencing_epoch = (
                next(iter(current_fencing_values))
                if len(current_fencing_values) == 1
                else fencing_epoch
            )
            current_records = _durable_docker_cleanup_bindings(
                profile,
                fencing_epoch=current_fencing_epoch,
                directory_anchor=(
                    cleanup_directory_anchor
                    if isinstance(
                        cleanup_directory_anchor,
                        _DurableCleanupDirectoryAnchor,
                    )
                    else None
                ),
            )
        except ValueError:
            cleanup_bindings_valid = False
            current_live_bindings = ()
            current_primary_members = current_tree.members
            current_records = ()
        cleanup_binding_set.update(current_live_bindings)
        cleanup_binding_set.update(
            record.binding for record in current_records
        )
        primary_fenced = bool(
            process.poll() is not None
            and not exact_root_alive
            and not current_primary_members
        )
        for record in current_records:
            watchdog_alive = _durable_cleanup_watchdog_alive(record)
            runner_alive = _durable_cleanup_runner_alive(record)
            if watchdog_alive is None or runner_alive is None:
                cleanup_bindings_valid = False
            elif (
                primary_fenced
                and runner_alive is False
                and watchdog_alive is False
            ):
                try:
                    create_state = (
                        "create_observed"
                        if record.termination_fence
                        else _durable_docker_create_state(record)
                    )
                    issuer_gone = (
                        _durable_docker_create_issuer_gone(record)
                        if create_state
                        in {"create_inflight", "create_outcome_unknown"}
                        else True
                    )
                except ValueError:
                    issuer_gone = None
                if issuer_gone is None:
                    cleanup_bindings_valid = False
                elif issuer_gone and not _reconcile_durable_docker_cleanup(
                    record,
                    deadline=deadline,
                ):
                    cleanup_bindings_valid = False
        try:
            remaining_records = _durable_docker_cleanup_bindings(
                profile,
                fencing_epoch=(current_fencing_epoch),
                directory_anchor=(
                    cleanup_directory_anchor
                    if isinstance(
                        cleanup_directory_anchor,
                        _DurableCleanupDirectoryAnchor,
                    )
                    else None
                ),
            )
        except ValueError:
            cleanup_bindings_valid = False
            remaining_records = current_records
        empty_now = bool(
            process.poll() is not None
            and not exact_root_alive
            and not current_tree.members
            and not remaining_records
        )
        if isinstance(daemon_kernel_fence, _ManagedDaemonKernelFence):
            if not isinstance(stored_identity, ProcessIdentity):
                empty_now = False
            else:
                strict_state, _strict_tree = (
                    _strict_plan_bound_process_fence_observation(
                        profile,
                        stored_identity,
                    )
                )
                empty_now = bool(empty_now and strict_state == "dead")
        stable_empty_samples = stable_empty_samples + 1 if empty_now else 0
        if stable_empty_samples >= 2:
            if (
                cleanup_bindings_valid
                and _detached_docker_cleanup_absence_verified(
                    tuple(sorted(cleanup_binding_set)),
                    deadline=deadline,
                )
            ):
                if not _release_process_cleanup_directory_anchor(process):
                    return False, member_pids
                return True, member_pids
        time.sleep(0.02)
    return False, member_pids


def stop_tracks(
    tracks: Sequence[SupervisorTrack],
    processes: dict[str, subprocess.Popen[bytes]],
    *,
    repo_root: Path,
    grace_seconds: float = 10.0,
    output: OutputFn = _default_output,
) -> dict[str, object]:
    """Stop exact marker-bound wrapper trees and verify no descendants remain.

    Lane shutdowns run concurrently so the bounded grace consumed by one slow
    or unverifiable tree cannot delay cooperative termination of another lane.
    Each worker still delegates exclusively to the existing lifecycle-profile
    and immutable process-birth checks in ``_terminate_managed_process``; an
    exception never becomes authority to signal a bare PID.
    """

    stopped: list[int] = []
    removed_runtime_markers: list[str] = []
    all_fenced = True
    _emit(output, "stopping supervisor wrapper and managed daemons")

    # Submit every exact managed tree before waiting for any one result.  The
    # previous serialized loop let one lane consume the caller's shutdown
    # window while later lanes had not even received cooperative termination.
    # A dedicated worker per bounded configured lane keeps the wall-clock
    # bound at the slowest lane rather than the sum of all lane grace periods.
    termination_results: dict[
        str,
        tuple[bool, tuple[int, ...], str],
    ] = {}
    managed = [
        (track, process)
        for track in tracks
        if (process := processes.get(track.name)) is not None
    ]
    if managed:
        with ThreadPoolExecutor(
            max_workers=len(managed),
            thread_name_prefix="agent-supervisor-stop",
        ) as executor:
            pending = [
                (
                    track,
                    executor.submit(
                        _terminate_managed_process,
                        process,
                        grace_seconds=grace_seconds,
                    ),
                )
                for track, process in managed
            ]
            for track, future in pending:
                try:
                    fenced, member_pids = future.result()
                except Exception as exc:
                    # Preserve fail-closed lifecycle semantics while allowing
                    # every independently identified lane to finish fencing.
                    termination_results[track.name] = (
                        False,
                        (),
                        type(exc).__name__,
                    )
                else:
                    termination_results[track.name] = (
                        bool(fenced),
                        tuple(member_pids),
                        "",
                    )

    for track in tracks:
        process = processes.get(track.name)
        fenced, member_pids, error_type = termination_results.get(
            track.name,
            (True, (), ""),
        )
        if fenced:
            stopped.extend(member_pids)
        elif process is not None:
            all_fenced = False
            error_suffix = (
                f" error_type={error_type}" if error_type else ""
            )
            _emit(
                output,
                (
                    "could not verify complete shutdown for "
                    f"{track.name} pid={process.pid}{error_suffix}"
                ),
            )
        if fenced and process is not None:
            resolved = track.resolve(repo_root)
            if _remove_stale_pid_marker_if_unchanged(
                resolved.supervisor_pid_path,
                process.pid,
            ):
                removed_runtime_markers.append(str(resolved.supervisor_pid_path))
            daemon_pid = read_pid_file(resolved.daemon_pid_path)
            if daemon_pid and _remove_stale_pid_marker_if_unchanged(
                resolved.daemon_pid_path,
                daemon_pid,
            ):
                removed_runtime_markers.append(str(resolved.daemon_pid_path))
    return {
        "stopped_pids": sorted(set(stopped)),
        "stopped_count": len(set(stopped)),
        "all_trees_fenced": all_fenced,
        "removed_runtime_markers": removed_runtime_markers,
    }


def _publish_plan_bound_terminal_missing(
    child: PlanBoundSupervisorChild,
    process: subprocess.Popen[bytes],
    *,
    repo_root: Path,
    reason_codes: Sequence[str],
) -> tuple[str, Any]:
    """Fence one exited current owner and terminally deny its whole wave."""

    from ..control.plan_execution_store import (
        ExecutionClaimConflictError,
        PlanBoundTerminalMissing,
        ProductionParallelPlanAdapter,
        _load_plan_bound_execution_lease_locked,
        _load_plan_bound_proposal_disposition_locked,
        _publish_plan_bound_terminal_missing_locked,
        _secure_store_cas,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    returncode = process.poll()
    profile = getattr(process, "_agent_supervisor_lifecycle_profile", None)
    process_identity = getattr(
        process,
        "_agent_supervisor_process_identity",
        None,
    )
    process_birth_cid = getattr(
        process,
        "_agent_supervisor_process_birth_cid",
        "",
    )
    if (
        returncode is None
        or not isinstance(profile, LifecycleProfile)
        or not isinstance(process_identity, ProcessIdentity)
        or not isinstance(process_birth_cid, str)
        or not process_birth_cid
    ):
        raise ExecutionClaimConflictError(
            "terminal-missing requires an exited durable process birth"
        )
    accepted_tree = _canonical_accepted_tree_root(Path(child.accepted_tree_root))
    resolved_repo = _canonical_accepted_tree_root(repo_root)
    if accepted_tree != resolved_repo:
        raise ExecutionClaimConflictError(
            "terminal-missing repository authority is mixed"
        )
    store_path = _lexical_contained_path(
        resolved_repo,
        _resolve_path(resolved_repo, Path(child.plan_revision_store_path)),
    )
    store = PlanRevisionStore(store_path)
    adapter = ProductionParallelPlanAdapter(store)
    with store._thread_lock:  # noqa: SLF001 - canonical one-winner transaction
        with store._guard():  # noqa: SLF001 - canonical cross-process guard
            if _load_plan_bound_proposal_disposition_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
            ) is not None:
                raise ExecutionClaimConflictError(
                    "terminal-missing conflicts with a proposal disposition"
                )
            execution_slice = adapter._validate_slice_owner_locked(  # noqa: SLF001
                revision_cid=child.revision_cid,
                slice_manifest_cid=child.slice_manifest_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
                reassignment_cid=child.reassignment_cid,
            )
            lease = _load_plan_bound_execution_lease_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
            )
            if (
                lease is None
                or lease[1].process_birth_cid != process_birth_cid
                or execution_slice.task_pairs
                != tuple(zip(child.task_ids, child.task_cids, strict=True))
            ):
                raise ExecutionClaimConflictError(
                    "terminal-missing lost its current execution lease"
                )
            process_state, fenced_tree = (
                _strict_plan_bound_process_fence_observation(
                    profile,
                    process_identity,
                )
            )
            if process_state != "dead" or fenced_tree is None:
                raise ExecutionClaimConflictError(
                    "terminal-missing process death is not provable"
                )
            observed_at_ms = int(time.time() * 1000)
            fence_record = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "plan-bound-terminal-process-fence@1"
                ),
                "revision_cid": child.revision_cid,
                "slice_manifest_cid": child.slice_manifest_cid,
                "slice_id": child.slice_id,
                "lane_id": child.lane_id,
                "reassignment_cid": child.reassignment_cid,
                "process_birth_cid": process_birth_cid,
                "profile": profile.to_dict(),
                "process_birth": process_identity.to_dict(),
                "fenced_tree": fenced_tree.to_dict(),
                "exit_code": int(returncode),
                "observed_at_ms": observed_at_ms,
            }
            process_fence_cid = store.put_cas(fence_record)
            if _secure_store_cas(store, process_fence_cid) != fence_record:
                raise ExecutionClaimConflictError(
                    "terminal-missing process fence failed CAS round trip"
                )
            terminal = PlanBoundTerminalMissing(
                revision_cid=child.revision_cid,
                plan_root_cid=child.plan_root_cid,
                execution_plan_cid=child.execution_plan_cid,
                capacity_snapshot_id=child.capacity_snapshot_id,
                slice_manifest_cid=child.slice_manifest_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
                reassignment_cid=child.reassignment_cid,
                task_id=child.task_ids[0],
                task_cid=child.task_cids[0],
                process_birth_cid=process_birth_cid,
                process_fence_cid=process_fence_cid,
                exit_code=int(returncode),
                observed_at_ms=observed_at_ms,
                reason_codes=tuple(reason_codes),
            )
            _publish_plan_bound_terminal_missing_locked(store, terminal)
            assignment = lease[1].assignment_for(
                child.task_ids[0],
                child.task_cids[0],
            )
            timeout_ms = assignment.get("lease_duration_ms")
            if (
                isinstance(timeout_ms, bool)
                or not isinstance(timeout_ms, int)
                or not 50 <= timeout_ms <= 86_400_000
            ):
                raise ExecutionClaimConflictError(
                    "terminal-missing compiled execution bound is invalid"
                )
            barrier = adapter._evaluate_wave_diff_barrier_locked(  # noqa: SLF001
                revision_cid=child.revision_cid,
                slice_manifest_cid=child.slice_manifest_cid,
                timeout_ms=timeout_ms,
                now_ms=observed_at_ms,
            )
            if barrier is None or barrier[1].decision != "missing":
                raise ExecutionClaimConflictError(
                    "terminal-missing did not deny the whole wave"
                )
            return barrier


def _plan_bound_process_birth_budget_reached(
    child: PlanBoundSupervisorChild,
) -> bool:
    """Return whether the current owner consumed its immutable birth budget."""

    from ..control.plan_execution_store import (
        MAX_PLAN_BOUND_WAVE_TRANSFERS,
        ProductionParallelPlanAdapter,
        _load_plan_bound_process_birth_chain_locked,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    accepted_tree = _canonical_accepted_tree_root(Path(child.accepted_tree_root))
    store_path = _lexical_contained_path(
        accepted_tree,
        _resolve_path(accepted_tree, Path(child.plan_revision_store_path)),
    )
    store = PlanRevisionStore(store_path)
    adapter = ProductionParallelPlanAdapter(store)
    with store._thread_lock:  # noqa: SLF001
        with store._guard():  # noqa: SLF001
            adapter._validate_slice_owner_locked(  # noqa: SLF001
                revision_cid=child.revision_cid,
                slice_manifest_cid=child.slice_manifest_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
                reassignment_cid=child.reassignment_cid,
            )
            binding = _load_plan_bound_process_birth_chain_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
            )
            if binding is None:
                raise ValueError("plan-bound recovery has no process-birth chain")
            return binding[1].generation == MAX_PLAN_BOUND_WAVE_TRANSFERS


def _publish_plan_bound_process_birth_exhausted(
    child: PlanBoundSupervisorChild,
    process: subprocess.Popen[bytes],
    *,
    repo_root: Path,
) -> tuple[str, Any]:
    """Fence the final recovery birth and durably require typed replanning."""

    from ..control.plan_execution_store import (
        MAX_PLAN_BOUND_WAVE_TRANSFERS,
        ExecutionClaimConflictError,
        PlanBoundProcessBirthExhausted,
        ProductionParallelPlanAdapter,
        _load_plan_bound_execution_lease_locked,
        _load_plan_bound_process_birth_chain_locked,
        _load_plan_bound_proposal_disposition_locked,
        _publish_plan_bound_process_birth_exhausted_locked,
        _secure_store_cas,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    returncode = process.poll()
    profile = getattr(process, "_agent_supervisor_lifecycle_profile", None)
    process_identity = getattr(
        process,
        "_agent_supervisor_process_identity",
        None,
    )
    process_birth_cid = getattr(
        process,
        "_agent_supervisor_process_birth_cid",
        "",
    )
    if (
        returncode is None
        or not isinstance(profile, LifecycleProfile)
        or not isinstance(process_identity, ProcessIdentity)
        or not isinstance(process_birth_cid, str)
        or not process_birth_cid
    ):
        raise ExecutionClaimConflictError(
            "process-birth exhaustion requires an exited durable birth"
        )
    accepted_tree = _canonical_accepted_tree_root(Path(child.accepted_tree_root))
    resolved_repo = _canonical_accepted_tree_root(repo_root)
    if accepted_tree != resolved_repo:
        raise ExecutionClaimConflictError(
            "process-birth exhaustion repository authority is mixed"
        )
    store_path = _lexical_contained_path(
        resolved_repo,
        _resolve_path(resolved_repo, Path(child.plan_revision_store_path)),
    )
    store = PlanRevisionStore(store_path)
    adapter = ProductionParallelPlanAdapter(store)
    with store._thread_lock:  # noqa: SLF001 - canonical one-winner transaction
        with store._guard():  # noqa: SLF001 - canonical cross-process guard
            execution_slice = adapter._validate_slice_owner_locked(  # noqa: SLF001
                revision_cid=child.revision_cid,
                slice_manifest_cid=child.slice_manifest_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
                reassignment_cid=child.reassignment_cid,
            )
            birth_binding = _load_plan_bound_process_birth_chain_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
            )
            if (
                birth_binding is None
                or birth_binding[0] != process_birth_cid
                or birth_binding[1].generation
                != MAX_PLAN_BOUND_WAVE_TRANSFERS
                or birth_binding[1].global_budget
                != MAX_PLAN_BOUND_WAVE_TRANSFERS
                or birth_binding[1].profile != profile.to_dict()
                or birth_binding[1].process_birth
                != process_identity.to_dict()
            ):
                raise ExecutionClaimConflictError(
                    "process-birth exhaustion lost the final chain head"
                )
            lease = _load_plan_bound_execution_lease_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
            )
            disposition = _load_plan_bound_proposal_disposition_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
            )
            if (
                lease is None
                or lease[1].phase
                not in {
                    "proposal_ready",
                    "merge_enqueue_prepared",
                    "merge_enqueue_confirmed",
                }
                or disposition is None
                or disposition[1].outcome not in {"changed", "no_change"}
                or execution_slice.task_pairs
                != tuple(zip(child.task_ids, child.task_cids, strict=True))
            ):
                raise ExecutionClaimConflictError(
                    "process-birth exhaustion lacks a recoverable disposition"
                )
            process_state, fenced_tree = (
                _strict_plan_bound_process_fence_observation(
                    profile,
                    process_identity,
                )
            )
            if (
                process_state != "dead"
                or fenced_tree is None
                or fenced_tree.members
            ):
                raise ExecutionClaimConflictError(
                    "process-birth exhaustion death is not provable"
                )
            observed_at_ms = int(time.time() * 1000)
            fence_record = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "plan-bound-process-birth-exhausted-fence@1"
                ),
                "revision_cid": child.revision_cid,
                "slice_manifest_cid": child.slice_manifest_cid,
                "slice_id": child.slice_id,
                "lane_id": child.lane_id,
                "reassignment_cid": child.reassignment_cid,
                "process_birth_cid": process_birth_cid,
                "generation": birth_binding[1].generation,
                "global_budget": birth_binding[1].global_budget,
                "profile": profile.to_dict(),
                "process_birth": process_identity.to_dict(),
                "fenced_tree": fenced_tree.to_dict(),
                "exit_code": int(returncode),
                "observed_at_ms": observed_at_ms,
            }
            process_fence_cid = store.put_cas(fence_record)
            if _secure_store_cas(store, process_fence_cid) != fence_record:
                raise ExecutionClaimConflictError(
                    "process-birth exhaustion fence failed CAS round trip"
                )
            terminal = PlanBoundProcessBirthExhausted(
                revision_cid=child.revision_cid,
                plan_root_cid=child.plan_root_cid,
                execution_plan_cid=child.execution_plan_cid,
                capacity_snapshot_id=child.capacity_snapshot_id,
                slice_manifest_cid=child.slice_manifest_cid,
                slice_id=child.slice_id,
                lane_id=child.lane_id,
                reassignment_cid=child.reassignment_cid,
                task_id=disposition[1].task_id,
                task_cid=disposition[1].task_cid,
                execution_lease_cid=lease[0],
                disposition_cid=disposition[0],
                process_birth_cid=process_birth_cid,
                process_fence_cid=process_fence_cid,
                generation=birth_binding[1].generation,
                global_budget=birth_binding[1].global_budget,
                exit_code=int(returncode),
                observed_at_ms=observed_at_ms,
                reason_codes=("process_birth_budget_exhausted",),
            )
            terminal_cid = _publish_plan_bound_process_birth_exhausted_locked(
                store,
                terminal,
            )
            return terminal_cid, terminal


def _plan_bound_child_has_disposition(
    child: PlanBoundSupervisorChild,
) -> bool:
    """Return whether the current slice owner published its one-winner result."""

    from ..control.plan_execution_store import (
        _load_plan_bound_proposal_disposition_locked,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    accepted_tree = _canonical_accepted_tree_root(Path(child.accepted_tree_root))
    store_path = _lexical_contained_path(
        accepted_tree,
        _resolve_path(accepted_tree, Path(child.plan_revision_store_path)),
    )
    store = PlanRevisionStore(store_path)
    with store._thread_lock:  # noqa: SLF001
        with store._guard():  # noqa: SLF001
            return _load_plan_bound_proposal_disposition_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
            ) is not None


def _plan_bound_child_execution_phase(
    child: PlanBoundSupervisorChild,
) -> str:
    """Load the current-owner execution phase through canonical authority."""

    from ..control.plan_execution_store import (
        ProductionParallelPlanAdapter,
        _load_plan_bound_merge_terminal_failure_locked,
        _load_plan_bound_process_birth_exhausted_locked,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    accepted_tree = _canonical_accepted_tree_root(Path(child.accepted_tree_root))
    store_path = _lexical_contained_path(
        accepted_tree,
        _resolve_path(accepted_tree, Path(child.plan_revision_store_path)),
    )
    store = PlanRevisionStore(store_path)
    with store._thread_lock:  # noqa: SLF001
        with store._guard():  # noqa: SLF001
            if _load_plan_bound_process_birth_exhausted_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
            ) is not None:
                return "process_birth_budget_exhausted"
            if _load_plan_bound_merge_terminal_failure_locked(
                store,
                revision_cid=child.revision_cid,
                slice_id=child.slice_id,
            ) is not None:
                return "merge_terminal_failure"
    adapter = ProductionParallelPlanAdapter(store)
    current = adapter.load_execution_lease(
        revision_cid=child.revision_cid,
        slice_id=child.slice_id,
        lane_id=child.lane_id,
    )
    return "" if current is None else current[1].phase


def _plan_bound_scope_drift_receipt(
    child: PlanBoundSupervisorChild,
) -> dict[str, Any] | None:
    """Read one typed pre-merge whole-wave denial through canonical authority."""

    from ..control.plan_execution_store import (
        ConfiguredBoardExecutionSlices,
        ProductionParallelPlanAdapter,
        _load_plan_bound_merge_terminal_failure_locked,
        _load_plan_bound_process_birth_exhausted_locked,
        _load_plan_bound_proposal_disposition_locked,
        _secure_store_cas,
    )
    from ..task_sources.plan_revision_store import PlanRevisionStore

    accepted_tree = _canonical_accepted_tree_root(
        Path(child.accepted_tree_root)
    )
    store_path = _lexical_contained_path(
        accepted_tree,
        accepted_tree / Path(child.plan_revision_store_path),
    )
    store = PlanRevisionStore(store_path)
    adapter = ProductionParallelPlanAdapter(store)
    terminal_rows: list[tuple[str, Mapping[str, Any]]] = []
    exhausted_rows: list[tuple[str, Any]] = []
    disposition_rows: list[tuple[str, Any]] = []
    with store._thread_lock:  # noqa: SLF001
        with store._guard():  # noqa: SLF001
            manifest = ConfiguredBoardExecutionSlices.from_dict(
                _secure_store_cas(store, child.slice_manifest_cid)
            )
            if manifest.plan_root_cid != child.plan_root_cid:
                raise ValueError(
                    "plan-bound terminal scan observed a foreign manifest"
                )
            for execution_slice in manifest.nonempty:
                terminal = _load_plan_bound_merge_terminal_failure_locked(
                    store,
                    revision_cid=child.revision_cid,
                    slice_id=execution_slice.slice_id,
                )
                if terminal is not None:
                    terminal_rows.append(terminal)
                exhausted = _load_plan_bound_process_birth_exhausted_locked(
                    store,
                    revision_cid=child.revision_cid,
                    slice_id=execution_slice.slice_id,
                )
                if exhausted is not None:
                    exhausted_rows.append(exhausted)
                disposition = _load_plan_bound_proposal_disposition_locked(
                    store,
                    revision_cid=child.revision_cid,
                    slice_id=execution_slice.slice_id,
                )
                if disposition is not None:
                    disposition_rows.append(disposition)
    if exhausted_rows:
        own = next(
            (
                record
                for _cid, record in disposition_rows
                if record.slice_id == child.slice_id
            ),
            None,
        )
        return {
            "kind": "process_birth_budget_exhausted",
            "decision": "missing",
            "revision_cid": child.revision_cid,
            "plan_root_cid": exhausted_rows[0][1].plan_root_cid,
            "slice_manifest_cid": child.slice_manifest_cid,
            "slice_id": child.slice_id,
            "lane_id": child.lane_id,
            "task_id": own.task_id if own is not None else child.task_ids[0],
            "task_cid": own.task_cid if own is not None else child.task_cids[0],
            "proposal_id": own.proposal_id if own is not None else "",
            "proposal_receipt_id": (
                own.proposal_receipt_id if own is not None else ""
            ),
            "reason_codes": ["process_birth_budget_exhausted"],
            "changed_paths": sorted(
                {
                    path
                    for _disposition_cid, disposition in disposition_rows
                    for path in disposition.actual_changed_paths
                }
            ),
            "merge_enqueue_reached": False,
            "process_birth_exhausted_cids": sorted(
                exhausted_cid for exhausted_cid, _record in exhausted_rows
            ),
        }
    if terminal_rows:
        own = next(
            (
                record
                for _cid, record in disposition_rows
                if record.slice_id == child.slice_id
            ),
            None,
        )
        return {
            "kind": "merge_terminal_failure",
            "decision": "merge_failed",
            "revision_cid": child.revision_cid,
            "plan_root_cid": terminal_rows[0][1]["plan_root_cid"],
            "slice_manifest_cid": child.slice_manifest_cid,
            "slice_id": child.slice_id,
            "lane_id": child.lane_id,
            "task_id": own.task_id if own is not None else child.task_ids[0],
            "task_cid": own.task_cid if own is not None else child.task_cids[0],
            "proposal_id": own.proposal_id if own is not None else "",
            "proposal_receipt_id": (
                own.proposal_receipt_id if own is not None else ""
            ),
            "reason_codes": sorted(
                {
                    reason
                    for _failure_cid, failure in terminal_rows
                    for reason in failure["reason_codes"]
                }
            ),
            "changed_paths": sorted(
                {
                    path
                    for _disposition_cid, disposition in disposition_rows
                    for path in disposition.actual_changed_paths
                }
            ),
            "merge_enqueue_reached": True,
            "merge_terminal_failure_cids": sorted(
                failure_cid for failure_cid, _failure in terminal_rows
            ),
        }
    barrier = adapter.load_wave_diff_barrier(
        revision_cid=child.revision_cid,
        slice_manifest_cid=child.slice_manifest_cid,
    )
    if barrier is not None and barrier[1].decision != "released":
        disposition_rows = []
        with store._thread_lock:  # noqa: SLF001
            with store._guard():  # noqa: SLF001
                for row in barrier[1].dispositions:
                    disposition = _load_plan_bound_proposal_disposition_locked(
                        store,
                        revision_cid=child.revision_cid,
                        slice_id=row["slice_id"],
                    )
                    if (
                        disposition is None
                        or disposition[0] != row["disposition_cid"]
                    ):
                        raise ValueError(
                            "wave barrier lost a disposition authority"
                        )
                    disposition_rows.append(disposition)
        own = next(
            (
                record
                for _cid, record in disposition_rows
                if record.slice_id == child.slice_id
            ),
            None,
        )
        return {
            "kind": "wave_diff_barrier",
            "wave_barrier_cid": barrier[0],
            "decision": barrier[1].decision,
            "revision_cid": barrier[1].revision_cid,
            "plan_root_cid": barrier[1].plan_root_cid,
            "slice_manifest_cid": barrier[1].slice_manifest_cid,
            "slice_id": child.slice_id,
            "lane_id": child.lane_id,
            "task_id": own.task_id if own is not None else child.task_ids[0],
            "task_cid": own.task_cid if own is not None else child.task_cids[0],
            "proposal_id": own.proposal_id if own is not None else "",
            "proposal_receipt_id": (
                own.proposal_receipt_id if own is not None else ""
            ),
            "reason_codes": list(barrier[1].reason_codes),
            "changed_paths": sorted(
                {
                    path
                    for _cid, record in disposition_rows
                    for path in record.actual_changed_paths
                }
            ),
            "merge_enqueue_reached": False,
        }
    execution_lease = adapter.load_execution_lease(
        revision_cid=child.revision_cid,
        slice_id=child.slice_id,
        lane_id=child.lane_id,
    )
    if execution_lease is None or execution_lease[1].phase != "scope_drift":
        return None
    drift = execution_lease[1]
    return {
        "kind": "legacy_scope_drift_lease",
        "execution_lease_cid": execution_lease[0],
        "revision_cid": drift.revision_cid,
        "plan_root_cid": drift.plan_root_cid,
        "slice_manifest_cid": drift.slice_manifest_cid,
        "slice_id": drift.slice_id,
        "lane_id": drift.lane_id,
        "task_id": drift.active_task_id,
        "task_cid": drift.active_task_cid,
        "proposal_id": drift.proposal_id,
        "proposal_receipt_id": drift.proposal_receipt_id,
        "reason_codes": list(drift.proposal_reason_codes),
        "changed_paths": list(drift.actual_changed_paths),
        "merge_enqueue_reached": drift.merge_enqueue_reached,
    }


def run_supervisor_tracks(
    tracks: Sequence[SupervisorTrack],
    *,
    repo_root: Path,
    common_args: Sequence[str],
    duration_seconds: float,
    heartbeat_interval_seconds: float = 60.0,
    supervisor_status_stale_seconds: float = 600.0,
    stop_grace_seconds: float = 10.0,
    python_executable: str = "python3",
    master_pid_path: Path | None = None,
    label: str = "multi-supervisor",
    exit_when_all_tracks_terminal: bool = False,
    plan_bound_children: Sequence[PlanBoundSupervisorChild] = (),
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
    require_configured_board_live_seal: str = "",
    output: OutputFn = _default_output,
) -> dict[str, object]:
    """Run and supervise multiple tracks for the requested duration."""

    managed_tracks = list(tracks)
    live_profile_required = _configured_board_live_seal_required(
        common_args,
        managed_tracks,
    )
    live_config = str(require_configured_board_live_seal or "")
    if live_config and not live_profile_required:
        raise ValueError(
            "configured-board live-seal flag requires the exact "
            "datasets-authoritative operational profile"
        )
    if live_config:
        relative = _configured_board_gate_relative_path(
            live_config,
            field="configured-board live-seal config",
        )
        if relative != CONFIGURED_BOARD_LIVE_SEAL_CONFIG_PATH:
            raise ValueError(
                "configured-board live seal requires the canonical scheduler config"
            )
        raise ValueError(CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO)
    resolved_repo_root = repo_root.resolve()
    plan_children_by_name = {
        child.name: child for child in plan_bound_children
    }
    if len(plan_children_by_name) != len(tuple(plan_bound_children)):
        raise ValueError("plan-bound child names must be unique")
    if plan_bound_children:
        if accepted_control_plane_pin is None:
            raise ValueError(
                "plan-bound wave requires a sealed accepted control plane"
            )
        verify_agent_implementation_sealed_control_plane(
            accepted_control_plane_pin,
            accepted_control_plane_descriptor,
        )
    initial_track_names = {track.name for track in managed_tracks}
    if not set(plan_children_by_name).issubset(initial_track_names):
        raise ValueError("every plan-bound child must own one launched track")
    lane_templates = {
        child.lane_id: child for child in plan_bound_children
    }
    if len(lane_templates) != len(tuple(plan_bound_children)):
        raise ValueError("plan-bound child lane IDs must be unique within a wave")
    resolved_master_pid: Path | None = None
    if master_pid_path is not None:
        resolved_master_pid = _resolve_path(resolved_repo_root, master_pid_path)
        resolved_master_pid.parent.mkdir(parents=True, exist_ok=True)
        if plan_bound_children:
            master_descriptor, master_identity = (
                _reserve_owned_pid_projection(resolved_master_pid)
            )
            try:
                _publish_reserved_pid_projection(
                    resolved_master_pid,
                    master_descriptor,
                    master_identity,
                    os.getpid(),
                )
            except BaseException:
                _discard_reserved_pid_projection(
                    resolved_master_pid,
                    master_identity,
                )
                raise
            finally:
                os.close(master_descriptor)
        else:
            _adopt_or_create_current_master_pid_projection(
                resolved_master_pid
            )
    processes: dict[str, subprocess.Popen[bytes]] = {}

    def _handle_signal(signum: int, _frame: object) -> None:
        raise SupervisorRunInterrupted(f"received signal {signum}")

    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    interrupted = ""
    blocked = ""
    terminal_quiescent = False
    bounded_finished_tracks: set[str] = set()
    pending_failed_slices: list[
        tuple[PlanBoundSupervisorChild, subprocess.Popen[bytes]]
    ] = []
    reassignment_count = 0
    reassignment_blockers: list[str] = []
    scope_drift_receipts: list[dict[str, Any]] = []
    replan_required = False
    run_started_at = time.time()
    track_generation_started_at: dict[str, float] = {}
    track_startup_grace_seconds: dict[str, float] = {}

    def start_generation(track: SupervisorTrack) -> subprocess.Popen[bytes]:
        """Start one track and retain the lower bound for its status generation."""

        startup_grace = _track_supervisor_status_startup_grace_seconds(
            track,
            common_args=common_args,
            fallback_seconds=float(supervisor_status_stale_seconds),
        )
        generation_started_at = time.time()
        process = start_track(
            track,
            repo_root=resolved_repo_root,
            common_args=common_args,
            python_executable=python_executable,
            accepted_control_plane_pin=accepted_control_plane_pin,
            accepted_control_plane_descriptor=(
                accepted_control_plane_descriptor
            ),
            output=output,
        )
        track_generation_started_at[track.name] = generation_started_at
        track_startup_grace_seconds[track.name] = startup_grace
        return process

    def recovery_recipient(
        donor: PlanBoundSupervisorChild,
    ) -> PlanBoundSupervisorChild:
        """Mint a fresh logical lane in the dead donor's freed process slot."""

        from ..control.plan_execution_store import ProductionParallelPlanAdapter
        from ..task_sources.plan_revision_store import PlanRevisionStore

        store_path = _lexical_contained_path(
            resolved_repo_root,
            _resolve_path(
                resolved_repo_root,
                Path(donor.plan_revision_store_path),
            ),
        )
        current = ProductionParallelPlanAdapter(
            PlanRevisionStore(store_path)
        ).load_slice_reassignment(
            revision_cid=donor.revision_cid,
            slice_id=donor.slice_id,
        )
        generation = current[1].generation + 1 if current is not None else 1
        token = hashlib.sha256(
            f"{donor.revision_cid}:{donor.slice_id}:{generation}".encode()
        ).hexdigest()[:12]
        lane_id = f"recovery-{generation}-{token}"
        state_parent = PurePosixPath(str(donor.state_dir)).parent
        return replace(
            donor,
            name=f"recovery-{generation}-{token}",
            state_dir=str(state_parent / lane_id),
            state_prefix=f"recovery_{generation}_{token}",
            lane_id=lane_id,
            reassignment_cid=donor.reassignment_cid,
        )

    def dispatch_pending_reassignments() -> None:
        nonlocal blocked, reassignment_count, replan_required
        while pending_failed_slices:
            donor, donor_process = pending_failed_slices.pop(0)
            selected_recipient: PlanBoundSupervisorChild | None = None
            try:
                selected_recipient = recovery_recipient(donor)
                adopted = reassign_fenced_plan_bound_child(
                    donor=donor,
                    recipient=selected_recipient,
                    donor_process=donor_process,
                    repo_root=resolved_repo_root,
                )
                adopted_track = adopted.track()
                if adopted_track.name in {track.name for track in managed_tracks}:
                    raise ValueError("reassigned track name is not unique")
                managed_tracks.append(adopted_track)
                plan_children_by_name[adopted.name] = adopted
                processes[adopted_track.name] = start_generation(adopted_track)
                reassignment_count += 1
                _emit(
                    output,
                    (
                        f"reassigned bounded slice={donor.slice_id} "
                        f"from_lane={donor.lane_id} "
                        f"to_lane={adopted.lane_id} "
                        f"reassignment_cid={adopted.reassignment_cid}"
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - typed fail-closed boundary
                blocker = (
                    f"slice={donor.slice_id} donor_lane={donor.lane_id} "
                    f"recipient_lane={getattr(selected_recipient, 'lane_id', '')} "
                    f"{type(exc).__name__}: {exc}"
                )
                reassignment_blockers.append(blocker)
                _emit(output, f"plan-bound reassignment blocked: {blocker}")
                try:
                    _publish_plan_bound_terminal_missing(
                        donor,
                        donor_process,
                        repo_root=resolved_repo_root,
                        reason_codes=(
                            "process_exited_without_disposition",
                            "safe_reassignment_exhausted",
                        ),
                    )
                    receipt = _plan_bound_scope_drift_receipt(donor)
                    if receipt is None:
                        raise ValueError(
                            "terminal-missing barrier receipt is absent"
                        )
                    scope_drift_receipts.append(receipt)
                    replan_required = True
                    blocked = (
                        "process-fenced missing slice requires a new plan revision"
                    )
                except Exception as terminal_exc:  # noqa: BLE001
                    terminal_blocker = (
                        f"slice={donor.slice_id} lane={donor.lane_id} "
                        f"terminal-missing {type(terminal_exc).__name__}: "
                        f"{terminal_exc}"
                    )
                    reassignment_blockers.append(terminal_blocker)
                    _emit(
                        output,
                        f"plan-bound terminal-missing blocked: {terminal_blocker}",
                    )

    try:
        _emit(output, f"starting {label} duration_seconds={duration_seconds:g}")
        for track in managed_tracks:
            processes[track.name] = start_generation(track)

        deadline = time.monotonic() + max(0.0, float(duration_seconds))
        while time.monotonic() < deadline:
            terminal_tracks: set[str] = set(bounded_finished_tracks)
            sleep_for = min(
                max(0.05, heartbeat_interval_seconds),
                max(0.0, deadline - time.monotonic()),
            )
            time.sleep(sleep_for)
            for track in tuple(managed_tracks):
                if track.name in bounded_finished_tracks:
                    continue
                process = processes.get(track.name)
                resolved = track.resolve(resolved_repo_root)
                daemon_fields = daemon_pid_health_fields(
                    resolved.daemon_pid_path,
                    cleanup_stale_marker=True,
                )
                supervisor_fields = supervisor_status_health_fields(
                    resolved,
                    repo_root=resolved_repo_root,
                    stale_seconds=float(supervisor_status_stale_seconds),
                    expected_supervisor_pid=(
                        None if process is None else int(process.pid)
                    ),
                    generation_started_at_epoch_seconds=(
                        track_generation_started_at.get(track.name)
                    ),
                    startup_grace_seconds=track_startup_grace_seconds.get(
                        track.name,
                        0.0,
                    ),
                )
                if process is not None and process.poll() is None and pid_alive(process.pid):
                    supervisor_summary = format_supervisor_status_fields(supervisor_fields)
                    heartbeat_parts = [
                        f"heartbeat {track.name} supervisor_pid={process.pid}",
                        format_daemon_heartbeat_fields(daemon_fields),
                    ]
                    if supervisor_summary:
                        heartbeat_parts.append(supervisor_summary)
                    _emit(
                        output,
                        " ".join(heartbeat_parts),
                    )
                    if supervisor_fields.get("restart_supervisor"):
                        if (
                            supervisor_fields.get(
                                "supervisor_status_generation_reason"
                            )
                            == "status_missing"
                        ):
                            # A live wrapper child with a one-sample empty
                            # status file is a torn heartbeat, not a dead
                            # generation. Restarting it SIGTERMs sibling
                            # lanes when the process tree cannot be fenced.
                            continue
                        daemon_pid = daemon_fields.get("daemon_pid")
                        _emit(
                            output,
                            (
                                f"restarting stale {track.name} supervisor old_pid={process.pid} "
                                f"daemon_pid={daemon_pid or 'unknown'} "
                                f"supervisor_status_age_seconds="
                                f"{supervisor_fields.get('supervisor_status_age_seconds')}"
                            ),
                        )
                        fenced, _member_pids = _terminate_managed_process(
                            process,
                            grace_seconds=stop_grace_seconds,
                        )
                        if not fenced:
                            raise SupervisorRunInterrupted(
                                f"could not fence stale {track.name} process tree"
                            )
                        try:
                            process.wait(timeout=max(0.1, stop_grace_seconds))
                        except subprocess.TimeoutExpired:
                            pass
                        processes[track.name] = start_generation(track)
                    elif exit_when_all_tracks_terminal:
                        task_fields = terminal_task_state_fields(
                            resolved,
                            repo_root=resolved_repo_root,
                            fresh_after_epoch_seconds=run_started_at,
                        )
                        if task_fields.get("terminal_quiescent"):
                            terminal_tracks.add(track.name)
                    continue
                old_pid = None if process is None else process.pid
                if "--plan-bound-dispatch" in track.extra_args:
                    returncode = None if process is None else process.poll()
                    if process is not None:
                        fenced, _member_pids = _terminate_managed_process(
                            process,
                            grace_seconds=stop_grace_seconds,
                        )
                        if not fenced:
                            raise SupervisorRunInterrupted(
                                f"could not fence completed {track.name} descendants"
                            )
                    plan_child = plan_children_by_name.get(track.name)
                    recover_execution = False
                    if plan_child is not None and process is not None:
                        scope_drift = None
                        authority_read_failed = False
                        try:
                            scope_drift = _plan_bound_scope_drift_receipt(
                                plan_child
                            )
                        except Exception as exc:  # noqa: BLE001 - authority boundary
                            blocker = (
                                "cannot read completed plan-bound execution lease: "
                                f"slice={plan_child.slice_id} "
                                f"lane={plan_child.lane_id} "
                                f"{type(exc).__name__}: {exc}"
                            )
                            reassignment_blockers.append(blocker)
                            blocked = blocker
                            authority_read_failed = True
                        if scope_drift is not None:
                            scope_drift_receipts.append(scope_drift)
                            replan_required = True
                            blocked = (
                                "typed actual candidate scope drift requires "
                                "a new serialized plan revision"
                            )
                        elif not authority_read_failed:
                            try:
                                has_disposition = (
                                    _plan_bound_child_has_disposition(plan_child)
                                )
                            except Exception as exc:  # noqa: BLE001
                                blocker = (
                                    "cannot prove completed plan-bound disposition: "
                                    f"slice={plan_child.slice_id} "
                                    f"lane={plan_child.lane_id} "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                reassignment_blockers.append(blocker)
                                blocked = blocker
                                authority_read_failed = True
                            if not authority_read_failed and not has_disposition:
                                if returncode not in (None, 0, 75):
                                    pending_failed_slices.append(
                                        (plan_child, process)
                                    )
                                else:
                                    try:
                                        _publish_plan_bound_terminal_missing(
                                            plan_child,
                                            process,
                                            repo_root=resolved_repo_root,
                                            reason_codes=(
                                                "process_exited_without_disposition",
                                            ),
                                        )
                                        receipt = _plan_bound_scope_drift_receipt(
                                            plan_child
                                        )
                                        if receipt is None:
                                            raise ValueError(
                                                "terminal-missing receipt is absent"
                                            )
                                        scope_drift_receipts.append(receipt)
                                        replan_required = True
                                        blocked = (
                                            "process-fenced missing slice requires "
                                            "a new plan revision"
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        blocker = (
                                            "cannot terminalize missing plan slice: "
                                            f"slice={plan_child.slice_id} "
                                            f"lane={plan_child.lane_id} "
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                        reassignment_blockers.append(blocker)
                                        blocked = blocker
                            elif not authority_read_failed:
                                try:
                                    execution_phase = (
                                        _plan_bound_child_execution_phase(
                                            plan_child
                                        )
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    blocker = (
                                        "cannot classify completed plan-bound "
                                        "handoff: "
                                        f"slice={plan_child.slice_id} "
                                        f"lane={plan_child.lane_id} "
                                        f"{type(exc).__name__}: {exc}"
                                    )
                                    reassignment_blockers.append(blocker)
                                    blocked = blocker
                                    authority_read_failed = True
                                else:
                                    recover_execution = execution_phase in {
                                        "proposal_ready",
                                        "merge_enqueue_prepared",
                                        "merge_enqueue_confirmed",
                                    }
                                    if (
                                        not recover_execution
                                        and execution_phase
                                        != "merge_completed"
                                    ):
                                        blocker = (
                                            "published disposition is not a "
                                            "terminal or recoverable handoff: "
                                            f"slice={plan_child.slice_id} "
                                            f"lane={plan_child.lane_id} "
                                            f"phase={execution_phase!r}"
                                        )
                                        reassignment_blockers.append(blocker)
                                        blocked = blocker
                    if (
                        recover_execution
                        and not blocked
                        and plan_child is not None
                        and process is not None
                    ):
                        try:
                            birth_budget_reached = (
                                _plan_bound_process_birth_budget_reached(
                                    plan_child
                                )
                            )
                        except Exception as exc:  # noqa: BLE001
                            blocker = (
                                "cannot validate recoverable process-birth budget: "
                                f"slice={plan_child.slice_id} "
                                f"lane={plan_child.lane_id} "
                                f"{type(exc).__name__}: {exc}"
                            )
                            reassignment_blockers.append(blocker)
                            blocked = blocker
                            birth_budget_reached = False
                        if birth_budget_reached and not blocked:
                            try:
                                _publish_plan_bound_process_birth_exhausted(
                                    plan_child,
                                    process,
                                    repo_root=resolved_repo_root,
                                )
                                receipt = _plan_bound_scope_drift_receipt(
                                    plan_child
                                )
                                if (
                                    receipt is None
                                    or receipt.get("kind")
                                    != "process_birth_budget_exhausted"
                                ):
                                    raise ValueError(
                                        "process-birth exhaustion receipt is absent"
                                    )
                                scope_drift_receipts.append(receipt)
                                replan_required = True
                                blocked = (
                                    "bounded plan-bound recovery births were "
                                    "exhausted; a new revision is required"
                                )
                                recover_execution = False
                            except Exception as exc:  # noqa: BLE001
                                blocker = (
                                    "cannot terminalize process-birth exhaustion: "
                                    f"slice={plan_child.slice_id} "
                                    f"lane={plan_child.lane_id} "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                reassignment_blockers.append(blocker)
                                blocked = blocker
                    if recover_execution and not blocked:
                        try:
                            processes[track.name] = start_generation(track)
                        except Exception as exc:  # noqa: BLE001
                            blocker = (
                                "cannot restart recoverable plan-bound handoff: "
                                f"slice={getattr(plan_child, 'slice_id', '')} "
                                f"lane={getattr(plan_child, 'lane_id', '')} "
                                f"{type(exc).__name__}: {exc}"
                            )
                            reassignment_blockers.append(blocker)
                            blocked = blocker
                        else:
                            _emit(
                                output,
                                (
                                    f"recovering bounded {track.name} "
                                    f"old_pid={old_pid or 'none'} "
                                    f"returncode={returncode!r}"
                                ),
                            )
                            continue
                    bounded_finished_tracks.add(track.name)
                    terminal_tracks.add(track.name)
                    _emit(
                        output,
                        (
                            f"completed bounded {track.name} supervisor "
                            f"old_pid={old_pid or 'none'} "
                            f"returncode={returncode!r}"
                        ),
                    )
                    continue
                _emit(output, f"restarting exited {track.name} supervisor old_pid={old_pid or 'none'}")
                if process is not None:
                    fenced, _member_pids = _terminate_managed_process(
                        process,
                        grace_seconds=stop_grace_seconds,
                    )
                    if not fenced:
                        raise SupervisorRunInterrupted(
                            f"could not fence exited {track.name} descendants"
                        )
                processes[track.name] = start_generation(track)
            dispatch_pending_reassignments()
            if replan_required:
                _emit(
                    output,
                    "fencing plan-bound wave for typed scope-drift STEER",
                )
                break
            if (
                exit_when_all_tracks_terminal
                and managed_tracks
                and len(terminal_tracks) == len(managed_tracks)
            ):
                if pending_failed_slices or reassignment_blockers:
                    blocked = (
                        "plan-bound wave ended with unreassigned failed slices"
                    )
                    _emit(
                        output,
                        (
                            f"blocked: {blocked} "
                            f"pending={len(pending_failed_slices)} "
                            f"reassignment_blockers={len(reassignment_blockers)}"
                        ),
                    )
                else:
                    terminal_quiescent = True
                    _emit(
                        output,
                        "all supervisor tracks reached fresh terminal quiescence",
                    )
                break
        if (
            plan_children_by_name
            and not terminal_quiescent
            and not replan_required
            and not blocked
            and any(
                name not in bounded_finished_tracks
                for name in plan_children_by_name
            )
        ):
            blocked = (
                "plan-bound wave exceeded its finite run window before "
                "every slice reached a terminal handoff"
            )
            _emit(output, f"blocked: {blocked}")
        if terminal_quiescent:
            _emit(output, "completed after terminal board drain")
        else:
            _emit(output, "completed requested run window")
    except PlanBoundProcessBirthError as exc:
        blocked = str(exc)
        _emit(
            output,
            (
                f"blocked: {blocked} pid={exc.pid} "
                f"profile_id={exc.profile_id} "
                f"all_trees_fenced={str(exc.all_trees_fenced).lower()}"
            ),
        )
    except SupervisorRunInterrupted as exc:
        interrupted = str(exc)
        _emit(output, f"interrupted: {interrupted}")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        stop_payload = stop_tracks(
            managed_tracks,
            processes,
            repo_root=resolved_repo_root,
            grace_seconds=stop_grace_seconds,
            output=output,
        )
        master_pid_removed = bool(
            resolved_master_pid is not None
            and stop_payload["all_trees_fenced"]
            and _remove_owned_pid_projection(resolved_master_pid, os.getpid())
        )
    return {
        "completed": not interrupted and not blocked,
        "interrupted": interrupted,
        "blocked": blocked,
        "track_count": len(managed_tracks),
        "reassignment_count": reassignment_count,
        "reassignment_blockers": reassignment_blockers,
        "unreassigned_failed_slice_count": len(pending_failed_slices),
        "stopped_count": stop_payload["stopped_count"],
        "all_trees_fenced": stop_payload["all_trees_fenced"],
        "removed_runtime_markers": stop_payload["removed_runtime_markers"],
        "master_pid_removed": master_pid_removed,
        "terminal_quiescent": terminal_quiescent,
        "replan_required": replan_required,
        "scope_drift_receipts": scope_drift_receipts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple implementation supervisors for a fixed window")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--duration-seconds", type=float, default=28800.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=60.0)
    parser.add_argument("--supervisor-status-stale-seconds", type=float, default=600.0)
    parser.add_argument("--stop-grace-seconds", type=float, default=10.0)
    parser.add_argument("--stamp", default=utc_run_stamp())
    parser.add_argument("--master-dir", type=Path, default=Path("data/agent_supervisor"))
    parser.add_argument("--master-log", type=Path, default=None)
    parser.add_argument("--master-pid-path", type=Path, default=None)
    parser.add_argument("--label", default="multi-supervisor")
    parser.add_argument(
        "--exit-when-all-tracks-terminal",
        action="store_true",
        help=(
            "End the run after every track publishes a fresh, complete, idle, "
            "unblocked task projection. Stale projections never trigger exit."
        ),
    )
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument(
        "--implementation-track",
        action="append",
        default=[],
        help="Compact NAME|SCRIPT|STATE_DIR|STATE_PREFIX implementation-supervisor track.",
    )
    parser.add_argument(
        "--implementation-plan-bound-track",
        action="append",
        default=[],
        help="Canonical JSON record for one exact nonempty plan-bound supervisor slice.",
    )
    parser.add_argument(
        "--plan-bound-wave",
        action="store_true",
        help="Run only the published nonempty slices, then return for coordinator replan.",
    )
    parser.add_argument(
        "--accepted-control-plane-pin-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--accepted-control-plane-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require-configured-board-live-seal",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--common-arg", action="append", default=[])
    parser.add_argument(
        "--implementation-supervisor-defaults",
        action="store_true",
        help="Prepend standard long-running implementation-supervisor args before --common-arg values.",
    )
    parser.add_argument("--implementation-supervisor-command", default="")
    parser.add_argument("--implementation-supervisor-stale-seconds", type=int, default=1800)
    parser.add_argument("--implementation-supervisor-check-interval", type=int, default=60)
    parser.add_argument("--implementation-supervisor-daemon-interval", type=int, default=120)
    parser.add_argument("--implementation-supervisor-timeout", type=int, default=1800)
    parser.add_argument("--implementation-supervisor-log-stall-seconds", type=int, default=900)
    parser.add_argument("--implementation-supervisor-max-restarts", type=int, default=0)
    parser.add_argument(
        "--implementation-supervisor-objective-scan-min-open-tasks",
        type=int,
        default=_env_int("OBJECTIVE_SCAN_MIN_OPEN_TASKS", 20),
    )
    parser.add_argument(
        "--implementation-supervisor-objective-scan-max-findings",
        type=int,
        default=_env_int("OBJECTIVE_SCAN_MAX_FINDINGS", 12),
    )
    parser.add_argument("--implementation-supervisor-objective-scan-cooldown-seconds", type=int, default=900)
    parser.add_argument(
        "--implementation-supervisor-objective-refill-timeout-seconds",
        type=int,
        default=_env_int("OBJECTIVE_REFILL_TIMEOUT_SECONDS", 600),
    )
    parser.add_argument(
        "--implementation-supervisor-objective-surplus-findings-per-goal",
        type=int,
        default=_env_int("OBJECTIVE_SURPLUS_FINDINGS_PER_GOAL", 6),
    )
    parser.add_argument(
        "--implementation-supervisor-objective-surplus-min-terms-per-todo",
        type=int,
        default=_env_int("OBJECTIVE_SURPLUS_MIN_TERMS_PER_TODO", 4),
    )
    parser.add_argument("--implementation-supervisor-codebase-scan-cooldown-seconds", type=int, default=900)
    parser.add_argument(
        "--implementation-supervisor-codebase-refill-timeout-seconds",
        type=int,
        default=_env_int("CODEBASE_REFILL_TIMEOUT_SECONDS", 600),
    )
    parser.add_argument("--implementation-supervisor-llm-merge-resolver-command", default="")
    parser.add_argument("--implementation-supervisor-llm-merge-resolver-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--implementation-supervisor-lanes-per-track",
        type=int,
        default=_env_int("IMPLEMENTATION_SUPERVISOR_LANES_PER_TRACK", 1),
        help=(
            "Launch N deterministic shard lanes for each implementation track. "
            "Each lane gets isolated state/worktree paths and task-shard args; merges remain serialized."
        ),
    )
    parser.add_argument(
        "--implementation-supervisor-strict-task-sharding",
        action="store_true",
        help=(
            "Disable cross-shard ready-task fallback in every implementation-supervisor "
            "lane, preventing lanes from borrowing the same retry work."
        ),
    )
    parser.add_argument(
        "--implementation-supervisor-idle-lane-work-stealing",
        choices=["virgin-transfer"],
        default="",
        help=(
            "Opt in every strict supervisor lane to exact-revision virgin "
            "task transfers."
        ),
    )
    parser.add_argument("--detach", action="store_true")
    return parser


def _master_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = args.repo_root.resolve()
    master_dir = _resolve_path(repo_root, args.master_dir)
    master_log = _resolve_path(repo_root, args.master_log) if args.master_log else master_dir / f"8h_run_{args.stamp}.log"
    master_pid = (
        _resolve_path(repo_root, args.master_pid_path)
        if args.master_pid_path
        else master_dir / f"8h_run_{args.stamp}.pid"
    )
    return master_log, master_pid


def _without_detach(argv: Sequence[str]) -> list[str]:
    removed = False
    cleaned: list[str] = []
    for item in argv:
        if item == "--detach" and not removed:
            removed = True
            continue
        cleaned.append(item)
    return cleaned


def _stream_targets_path(stream: _SupportsFileno, path: Path) -> bool:
    """Return whether a writable stream and path identify the same file."""

    try:
        stream_stat = os.fstat(stream.fileno())
        path_stat = path.stat()
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return (stream_stat.st_dev, stream_stat.st_ino) == (
        path_stat.st_dev,
        path_stat.st_ino,
    )


def launch_detached(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, object]:
    """Launch this runner detached, redirecting output to the master log."""

    if args.require_configured_board_live_seal:
        raise ValueError(CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO)

    master_log, master_pid = _master_paths(args)
    master_log.parent.mkdir(parents=True, exist_ok=True)
    master_pid.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
        *_without_detach(argv),
    ]
    from .process_security import (
        STATE_AUTHORITY_PARENT_LOSS_DETACHED,
        prepare_state_authority_child_handoff,
    )

    launch_environment = dict(os.environ)
    authority_handoff = prepare_state_authority_child_handoff(
        launch_environment,
        parent_loss_policy=STATE_AUTHORITY_PARENT_LOSS_DETACHED,
    )
    authority_descriptors = authority_handoff.pass_fds
    process: subprocess.Popen[bytes] | None = None
    descriptor = -1
    reservation_identity: tuple[int, int] | None = None
    with serialized_lock_update(master_pid):
        try:
            os.lstat(master_pid)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError("cannot inspect detached master PID projection") from exc
        else:
            _quarantine_stale_detached_master_pid_locked(master_pid)
        descriptor, reservation_identity = _reserve_owned_pid_projection_locked(
            master_pid
        )
        try:
            out_handle = master_log.open("ab")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=args.repo_root,
                    env=launch_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=out_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=authority_descriptors,
                )
                authority_handoff.deliver(process)
            finally:
                out_handle.close()
            _publish_reserved_pid_projection(
                master_pid,
                descriptor,
                reservation_identity,
                int(process.pid),
            )
        except BaseException:
            authority_handoff.close()
            if process is not None and process.poll() is None:
                try:
                    os.killpg(int(process.pid), signal.SIGTERM)
                    process.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(int(process.pid), signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
            _discard_reserved_pid_projection_locked(
                master_pid,
                reservation_identity,
            )
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
    assert process is not None
    # The child normally removes its own projection after fencing every
    # track.  Cover the short-run race where it exits before this parent can
    # publish the detached PID.
    if process.poll() is not None or not pid_alive(process.pid):
        _remove_owned_pid_projection(master_pid, process.pid)
    return {
        "stamp": args.stamp,
        "master_pid": process.pid,
        "master_log": str(master_log),
        "master_pid_file": str(master_pid),
    }


def common_args_from_parsed_args(args: argparse.Namespace) -> list[str]:
    """Return the effective common supervisor args for parsed runner options."""

    common_args: list[str] = []
    if args.implementation_supervisor_defaults:
        common_args.extend(
            implementation_supervisor_common_args(
                implementation_command=args.implementation_supervisor_command,
                llm_merge_resolver_command=(
                    args.implementation_supervisor_llm_merge_resolver_command
                ),
                stale_seconds=args.implementation_supervisor_stale_seconds,
                check_interval=args.implementation_supervisor_check_interval,
                daemon_interval=args.implementation_supervisor_daemon_interval,
                implementation_timeout=args.implementation_supervisor_timeout,
                implementation_log_stall_seconds=args.implementation_supervisor_log_stall_seconds,
                max_restarts=args.implementation_supervisor_max_restarts,
                objective_scan_min_open_tasks=args.implementation_supervisor_objective_scan_min_open_tasks,
                objective_scan_max_findings=args.implementation_supervisor_objective_scan_max_findings,
                objective_scan_cooldown_seconds=args.implementation_supervisor_objective_scan_cooldown_seconds,
                objective_refill_timeout_seconds=args.implementation_supervisor_objective_refill_timeout_seconds,
                objective_surplus_findings_per_goal=args.implementation_supervisor_objective_surplus_findings_per_goal,
                objective_surplus_min_terms_per_todo=args.implementation_supervisor_objective_surplus_min_terms_per_todo,
                codebase_scan_cooldown_seconds=args.implementation_supervisor_codebase_scan_cooldown_seconds,
                codebase_refill_timeout_seconds=args.implementation_supervisor_codebase_refill_timeout_seconds,
                llm_merge_resolver_timeout_seconds=args.implementation_supervisor_llm_merge_resolver_timeout_seconds,
                strict_task_sharding=bool(
                    getattr(
                        args,
                        "implementation_supervisor_strict_task_sharding",
                        False,
                    )
                ),
                idle_lane_work_stealing=str(
                    getattr(
                        args,
                        "implementation_supervisor_idle_lane_work_stealing",
                        "",
                    )
                    or ""
                ),
            )
        )
    if (
        bool(
            getattr(
                args,
                "implementation_supervisor_strict_task_sharding",
                False,
            )
        )
        and "--strict-task-sharding" not in common_args
    ):
        common_args.append("--strict-task-sharding")
    work_stealing = str(
        getattr(
            args,
            "implementation_supervisor_idle_lane_work_stealing",
            "",
        )
        or ""
    )
    if work_stealing and "--idle-lane-work-stealing" not in common_args:
        common_args.extend(["--idle-lane-work-stealing", work_stealing])
    common_args.extend(args.common_arg)
    return common_args


def tracks_from_parsed_args(args: argparse.Namespace) -> list[SupervisorTrack]:
    """Return supervisor tracks from raw and compact parsed track specs."""

    tracks = [parse_track_spec(track, stamp=args.stamp) for track in args.track]
    for track in args.implementation_track:
        tracks.extend(
            expand_implementation_track_lanes(
                track,
                stamp=args.stamp,
                lanes_per_track=args.implementation_supervisor_lanes_per_track,
            )
        )
    for record in getattr(args, "implementation_plan_bound_track", ()):
        tracks.append(PlanBoundSupervisorChild.from_cli_record(record).track(stamp=args.stamp))
    return tracks


def _run_plan_bound_launch_gate(argv: Sequence[str]) -> int:
    """Retain the exact gate interpreter through its final sealed exec."""

    tokens = tuple(str(item) for item in argv)
    if len(tokens) < 11 or tokens[8] != "--":
        return 78
    try:
        retained_interpreter = admit_retained_control_plane_interpreter(
            descriptor=int(tokens[5]),
            argv0=tokens[6],
            expected_sha256=tokens[7],
        )
    except (OSError, ValueError):
        return 78
    try:
        running = os.stat("/proc/self/exe")
        retained = os.fstat(retained_interpreter.descriptor)
        if (running.st_dev, running.st_ino) != (
            retained.st_dev,
            retained.st_ino,
        ):
            return 78
        os.set_inheritable(retained_interpreter.descriptor, False)
        return _run_plan_bound_launch_gate_with_interpreter(
            tokens,
            retained_interpreter=retained_interpreter,
        )
    except OSError:
        return 78
    finally:
        try:
            os.close(retained_interpreter.descriptor)
        except OSError:
            pass


def _run_plan_bound_launch_gate_with_interpreter(
    argv: Sequence[str],
    *,
    retained_interpreter: RetainedControlPlaneInterpreter,
) -> int:
    """Release exactly one accepted-tree child after parent birth capture."""

    tokens = tuple(str(item) for item in argv)
    if len(tokens) < 11 or tokens[8] != "--":
        return 78
    try:
        gate_fd = int(tokens[0])
        control_plane_pin = parse_accepted_control_plane_pin(tokens[2])
        control_plane_descriptor = int(tokens[3])
        recovery_authorization_cid = tokens[4]
        verify_agent_implementation_sealed_control_plane(
            control_plane_pin,
            control_plane_descriptor,
        )
    except ValueError:
        return 78
    try:
        accepted_tree_root = _canonical_accepted_tree_root(Path(tokens[1]))
    except ValueError:
        return 78
    child_command = list(tokens[9:])
    try:
        native_dependency, system_directories = (
            admit_sealed_native_dependency_environment(os.environ)
        )
        if (
            retained_interpreter.sha256
            != native_dependency.pin.python_executable_sha256
        ):
            raise ValueError("gate interpreter differs from native pin")
        expected_prefix = build_sealed_control_plane_module_command(
            python_executable=retained_interpreter.argv0,
            pin=control_plane_pin,
            descriptor=control_plane_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=(),
            retained_interpreter=retained_interpreter,
            native_dependency_launch=native_dependency,
            accepted_native_authorization_id=(
                native_dependency.accepted_authorization_id
            ),
            system_dependency_directories_json=system_directories,
        )
    except (IndexError, OSError, ValueError):
        return 78
    prefix_length = len(expected_prefix)
    if child_command[:prefix_length] != expected_prefix:
        return 78
    child_argv = child_command[prefix_length:]
    try:
        source_heads = _profile_option_values(
            child_argv,
            "--plan-bound-source-head",
        )
        source_trees = _profile_option_values(
            child_argv,
            "--plan-bound-source-tree",
        )
        child_roots = _profile_option_values(
            child_argv,
            "--plan-bound-accepted-tree-root",
        )
        store_paths = _profile_option_values(
            child_argv,
            "--plan-revision-store-path",
        )
        revision_cids = _profile_option_values(
            child_argv,
            "--plan-bound-revision-cid",
        )
        slice_ids = _profile_option_values(
            child_argv,
            "--plan-bound-slice-id",
        )
        lane_ids = _profile_option_values(
            child_argv,
            "--plan-bound-lane-id",
        )
        state_dirs = _profile_option_values(child_argv, "--state-dir")
        worktree_roots = _profile_option_values(child_argv, "--worktree-root")
        merge_queue_roots = _profile_option_values(
            child_argv,
            "--merge-queue-dir",
        )
    except ValueError:
        return 78
    if (
        gate_fd < 3
        or control_plane_descriptor < 3
        or gate_fd == control_plane_descriptor
        or "--plan-bound-dispatch" not in child_argv
        or child_roots != (str(accepted_tree_root),)
        or len(store_paths) != 1
        or len(revision_cids) != 1
        or len(slice_ids) != 1
        or len(lane_ids) != 1
        or len(source_heads) != 1
        or len(source_trees) != 1
        or not recovery_authorization_cid
        or (
            source_heads[0],
            source_trees[0],
        )
        != (
            control_plane_pin.source_head,
            control_plane_pin.source_tree,
        )
    ):
        return 78
    try:
        while True:
            try:
                authorization = os.read(gate_fd, 1)
                break
            except InterruptedError:
                continue
    except OSError:
        return 78
    finally:
        try:
            os.close(gate_fd)
        except OSError:
            pass
    if authorization != PLAN_BOUND_LAUNCH_GATE_SUCCESS:
        return 78
    try:
        recovery_repository_head = ""
        recovery_repository_tree = ""
        recovery_runtime_roots: tuple[Path, ...] = ()
        recovery_owner_bound_artifacts: tuple[Path, ...] = ()
        recovery_artifacts: tuple[Mapping[str, Any], ...] = ()
        if recovery_authorization_cid != "-":
            from ..control.plan_execution_store import (
                ProductionParallelPlanAdapter,
            )
            from ..task_sources.plan_revision_store import PlanRevisionStore

            store_path = _resolve_path(
                accepted_tree_root,
                Path(store_paths[0]),
            )
            _lexical_contained_path(accepted_tree_root, store_path)
            if (
                len(state_dirs) != 1
                or len(worktree_roots) != 1
                or len(merge_queue_roots) != 1
            ):
                return 78
            state_dir = _resolve_path(
                accepted_tree_root,
                Path(state_dirs[0]),
            )
            if state_dir.parent != store_path.parent:
                return 78
            recovery_runtime_roots = (
                store_path.parent,
                _resolve_path(
                    accepted_tree_root,
                    Path(worktree_roots[0]),
                ),
                _resolve_path(
                    accepted_tree_root,
                    Path(merge_queue_roots[0]),
                ),
            )
            plan_adapter = ProductionParallelPlanAdapter(
                PlanRevisionStore(store_path)
            )
            recovery = plan_adapter.load_recovery_launch(
                revision_cid=revision_cids[0],
                slice_id=slice_ids[0],
                lane_id=lane_ids[0],
                authorization_cid=recovery_authorization_cid,
            )
            execution = plan_adapter.load_execution_lease(
                revision_cid=revision_cids[0],
                slice_id=slice_ids[0],
                lane_id=lane_ids[0],
            )
            if (
                recovery.source_head != source_heads[0]
                or recovery.source_tree != source_trees[0]
                or execution is None
                or execution[0] != recovery.execution_lease_cid
            ):
                return 78
            recovery_repository_head = recovery.repository_head
            recovery_repository_tree = recovery.repository_tree
            recovery_artifacts = recovery.runtime_artifacts
            recovery_owner_bound_artifacts = (
                state_dir / "implementation.lock",
                *(
                    _resolve_path(accepted_tree_root, Path(path))
                    for path in plan_adapter.recovery_workspace_paths(
                        revision_cid=revision_cids[0],
                        slice_manifest_cid=recovery.slice_manifest_cid,
                    )
                ),
                *(
                    _resolve_path(accepted_tree_root, Path(path))
                    for path in recovery.launch_artifact_paths
                ),
            )
        _validate_plan_bound_accepted_tree(
            accepted_tree_root=accepted_tree_root,
            source_head=source_heads[0],
            source_tree=source_trees[0],
            control_plane_pin=control_plane_pin,
            recovery_repository_head=recovery_repository_head,
            recovery_repository_tree=recovery_repository_tree,
            recovery_runtime_roots=recovery_runtime_roots,
            recovery_owner_bound_artifacts=(
                recovery_owner_bound_artifacts
            ),
            recovery_artifacts=recovery_artifacts,
        )
    except (
        OSError,
        UnicodeError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
        return 78
    try:
        from .process_security import STATE_AUTHORITY_HANDOFF_ENV_NAMES

        environment = _plan_bound_positive_child_environment(os.environ)
        environment.update(
            {
                name: str(os.environ[name])
                for name in STATE_AUTHORITY_HANDOFF_ENV_NAMES
                if str(os.environ.get(name, "") or "").strip()
            }
        )
        environment.update(
            sealed_native_dependency_environment(
                native_dependency,
                system_dependency_directories_json=system_directories,
            )
        )
        os.execve(
            retained_interpreter.executable_path,
            child_command,
            environment,
        )
    except OSError:
        return 78
    return 78


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list[:1] == [CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_GATE_MARKER]:
        return 78
    if args_list[:1] == [PLAN_BOUND_LAUNCH_GATE_MARKER]:
        # The accepted-tree gate must remain authority-free.  It preserves the
        # one-shot handoff environment across its final exec; the sealed target
        # hardens and redeems the descriptor only after that exec boundary.
        return _run_plan_bound_launch_gate(args_list[1:])
    from .process_security import harden_state_authority_process

    harden_state_authority_process()
    parser = build_arg_parser()
    args = parser.parse_args(args_list)
    if (
        not args.track
        and not args.implementation_track
        and not args.implementation_plan_bound_track
        and not args.plan_bound_wave
    ):
        parser.error("at least one --track or --implementation-track is required")
    if args.require_configured_board_live_seal:
        parser.error(CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO)
    if (
        args.implementation_track
        or args.implementation_plan_bound_track
        or args.implementation_supervisor_defaults
    ):
        try:
            seal_ordered_implementation_provider_route(
                repo_root=args.repo_root,
            )
        except ValueError as exc:
            parser.error(str(exc))
        # Fail closed before leasing worktrees when the Grok/Codex entry module
        # is missing provider-command symbols; heal known gaps automatically.
        try:
            from .provider_command_binding import (
                ProviderCommandBindingError,
                preflight_provider_entry_module,
            )

            preflight_provider_entry_module(
                "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
            )
        except ProviderCommandBindingError as exc:
            parser.error(f"provider command binding preflight failed: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface import failures as preflight
            parser.error(
                "provider entry module preflight failed: "
                f"{type(exc).__name__}: {exc}"
            )
    if args.detach:
        payload = launch_detached(args, args_list)
        for key in ("stamp", "master_pid", "master_log", "master_pid_file"):
            print(f"{key}={payload[key]}")
        return 0

    if args.plan_bound_wave and not (
        args.track or args.implementation_track or args.implementation_plan_bound_track
    ):
        print("plan-bound wave has no nonempty slices", flush=True)
        return 0
    master_log, master_pid = _master_paths(args)
    plan_bound_children = tuple(
        PlanBoundSupervisorChild.from_cli_record(record)
        for record in getattr(args, "implementation_plan_bound_track", ())
    )
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None
    if plan_bound_children:
        try:
            accepted_control_plane_pin = parse_accepted_control_plane_pin(
                args.accepted_control_plane_pin_json
            )
            verify_agent_implementation_sealed_control_plane(
                accepted_control_plane_pin,
                args.accepted_control_plane_fd,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"sealed accepted control plane is invalid: {exc}")
        generations = {
            (child.source_head, child.source_tree)
            for child in plan_bound_children
        }
        if generations != {
            (
                accepted_control_plane_pin.source_head,
                accepted_control_plane_pin.source_tree,
            )
        }:
            parser.error(
                "plan-bound slices differ from the accepted control-plane generation"
            )
    tracks = tracks_from_parsed_args(args)
    master_log.parent.mkdir(parents=True, exist_ok=True)
    with master_log.open("ab") as log_handle:
        stdout_is_master_log = _stream_targets_path(sys.stdout, master_log)

        def output(message: str) -> None:
            print(message, flush=True)
            if not stdout_is_master_log:
                log_handle.write((message + "\n").encode("utf-8"))
                log_handle.flush()

        run_result = run_supervisor_tracks(
            tracks,
            repo_root=args.repo_root,
            common_args=common_args_from_parsed_args(args),
            duration_seconds=args.duration_seconds,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            supervisor_status_stale_seconds=args.supervisor_status_stale_seconds,
            stop_grace_seconds=args.stop_grace_seconds,
            python_executable=args.python_executable,
            master_pid_path=master_pid,
            label=args.label,
            exit_when_all_tracks_terminal=(
                args.exit_when_all_tracks_terminal or args.plan_bound_wave
            ),
            plan_bound_children=plan_bound_children,
            accepted_control_plane_pin=accepted_control_plane_pin,
            accepted_control_plane_descriptor=args.accepted_control_plane_fd,
            output=output,
        )
    if (
        args.plan_bound_wave
        and run_result.get("replan_required") is True
        and run_result.get("all_trees_fenced") is True
    ):
        return PLAN_BOUND_REPLAN_RETURN_CODE
    if args.plan_bound_wave and (
        run_result.get("completed") is not True
        or run_result.get("all_trees_fenced") is not True
    ):
        return 2
    return 0


CASF_EVENT_DRIVEN_WAKE_INTERFACE = (
    "ipfs_accelerate_py/agent-supervisor/casf-event-driven-wake@1"
)


def casf_require_event_driven_wait(wait_capability: Mapping[str, object]) -> None:
    """Fail closed unless the federation wait path is event-driven qualified."""

    from ipfs_accelerate_py.agent_supervisor.federation.scheduler import (
        require_event_driven_capability,
    )

    require_event_driven_capability(wait_capability)


def casf_select_tracks_for_frontier(
    tracks: Sequence[SupervisorTrack],
    *,
    must_wake: Sequence[str],
    may_wake: Sequence[str],
    do_not_wake: Sequence[str],
    wait_capability: Mapping[str, object],
) -> tuple[SupervisorTrack, ...]:
    """Wake only frontier-eligible tracks. Unchanged do_not_wake tracks stay asleep."""

    casf_require_event_driven_wait(wait_capability)
    asleep = set(do_not_wake)
    eligible = set(must_wake) | set(may_wake)
    overlap = eligible & asleep
    if overlap:
        raise ValueError("frontier dispositions overlap")
    selected: list[SupervisorTrack] = []
    for track in tracks:
        if track.name in asleep:
            continue
        if track.name in eligible:
            selected.append(track)
    return tuple(selected)


if __name__ == "__main__":
    raise SystemExit(main())
