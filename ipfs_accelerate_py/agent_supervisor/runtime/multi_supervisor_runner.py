"""Reusable timed runner for multiple implementation supervisor scripts."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, MutableMapping, Sequence
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
_LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_FLAG = (
    "--require-lgcvf-configured-board-live-seal"
)
_LGCVF_CONFIGURED_BOARD_LIVE_BIRTH_MARKER = (
    "--run-lgcvf-configured-board-live-capsule-gate"
)
if __package__ in {None, ""} and (
    _CONFIGURED_BOARD_LIVE_LAUNCH_FLAG in sys.argv[1:]
    or _CONFIGURED_BOARD_LIVE_BIRTH_MARKER in sys.argv[1:]
    or _LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_FLAG in sys.argv[1:]
    or _LGCVF_CONFIGURED_BOARD_LIVE_BIRTH_MARKER in sys.argv[1:]
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
    build_agent_implementation_control_plane_pin,
    load_agent_implementation_route_authorization,
    resolve_agent_implementation_route,
    verify_agent_implementation_sealed_control_plane,
)
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
)
from ..core.wrapper_utils import (
    AgentSupervisorNamespacePaths,
    apply_env_defaults,
    env_str,
)
from ..merge.checkout_lock import serialized_lock_update
from ..proof.formal_verification_contracts import content_identity
from ..todo_daemon.core import pid_alive, read_pid_file, remove_runtime_marker

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
LGCVF_CONFIGURED_BOARD_LIVE_ADMISSION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "lgcvf-configured-board-live-admission@1"
)
LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH = (
    "config/agent_supervisor_logic_governed_compositional_verification_"
    "fabric_quack_candidate_scheduler.json"
)
LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE = (
    "logic-governed-compositional-verification-fabric-v1"
)
LGCVF_CONFIGURED_BOARD_LIVE_SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "logic_governed_compositional_verification_fabric.scheduler_config@1"
)
LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES = tuple(
    f"lgcvf-quack-lane-{index}" for index in range(4)
)
LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_MARKER = (
    _LGCVF_CONFIGURED_BOARD_LIVE_BIRTH_MARKER
)
LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_SUCCESS = b"\x01"
LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner"
)
LGCVF_CONFIGURED_BOARD_LIVE_MODULES = frozenset(
    {
        "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
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
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
    }
)
SEALED_CONTROL_PLANE_BOOTSTRAP = r'''import fcntl,hashlib,json,os,stat,sys
def _pairs(items):
    result={}
    for key,value in items:
        if key in result: raise SystemExit(78)
        result[key]=value
    return result
try:
    fd=int(sys.argv.pop(1)); pin=json.loads(sys.argv.pop(1),object_pairs_hook=_pairs)
    module=sys.argv.pop(1); expected_bootstrap=sys.argv.pop(1); expected_python=sys.argv.pop(1)
    if fd<3 or type(pin) is not dict or set(pin)!={'schema','runner_path','runner_sha256','capsule_root','capsule_id','source_head','source_tree','archive_sha256'}: raise SystemExit(78)
    if any(type(value) is not str or not value for value in pin.values()): raise SystemExit(78)
    if pin['schema']!='ipfs_accelerate_py.agent_supervisor.accepted-control-plane@2': raise SystemExit(78)
    if module not in {'ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler','ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'}: raise SystemExit(78)
    if module in {'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'}: raise SystemExit(78)
    command_line=open('/proc/self/cmdline','rb').read().split(b'\0')
    code_index=command_line.index(b'-c')+1
    if 'sha256:'+hashlib.sha256(command_line[code_index]).hexdigest()!=expected_bootstrap: raise SystemExit(78)
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
    sys.path.insert(0,archive)
    import importlib,importlib.machinery,runpy,types
    import ipfs_accelerate_py as accepted_root
    prefix=archive+'/'
    root_origin=getattr(accepted_root,'__file__',None)
    if type(root_origin) is not str or not root_origin.startswith(prefix): raise SystemExit(78)
    package_name='ipfs_accelerate_py.agent_supervisor'
    if any(name==package_name or name.startswith(package_name+'.') for name in sys.modules): raise SystemExit(78)
    package=importlib.import_module(package_name)
    package_origin=getattr(package,'__file__',None)
    if type(package_origin) is not str or not package_origin.startswith(prefix): raise SystemExit(78)
    setattr(accepted_root,'agent_supervisor',package)
    if module in sys.modules: raise SystemExit(78)
    namespace=runpy.run_module(module,run_name=module,alter_sys=True)
    if module in sys.modules: raise SystemExit(78)
    target_origin=namespace.get('__file__')
    if type(target_origin) is not str or not target_origin.startswith(prefix): raise SystemExit(78)
    for name,loaded in tuple(sys.modules.items()):
        if name=='ipfs_accelerate_py' or name=='ipfs_accelerate_py.llm_router' or name.startswith('ipfs_accelerate_py.agent_supervisor'):
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
LGCVF_CONFIGURED_BOARD_LIVE_BOOTSTRAP = r'''import ctypes,os,sys
def _deny(): raise SystemExit(78)
def _pairs(items):
    result={}
    for key,value in items:
        if key in result: _deny()
        result[key]=value
    return result
def _decode(value):
    try: result=json.loads(value,object_pairs_hook=_pairs)
    except BaseException: _deny()
    if type(result) is not dict or json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)!=value: _deny()
    return result
def _sha(value): return 'sha256:'+hashlib.sha256(value).hexdigest()
def _field(value,name):
    if type(value) is dict: return value.get(name)
    return getattr(value,name,None)
try:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode or not sys.platform.startswith('linux') or any(name.startswith(('LD_','PYTHON')) or name=='GLIBC_TUNABLES' for name in os.environ): _deny()
    if os.environ.get('IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY','')!='load_only': _deny()
    token=os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN','')
    try: token_bytes=token.encode('ascii')
    except BaseException: _deny()
    if not 8<=len(token_bytes)<=4096 or any(value not in b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for value in token_bytes): _deny()
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(4,0,0,0,0)!=0 or libc.prctl(3,0,0,0,0)!=0: _deny()
    import fcntl,hashlib,json,re,stat
    token_sink=os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE',''); token_marker=os.path.dirname(token_sink)
    try:
        marker_fd=os.open(token_marker,os.O_RDONLY|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0)); marker_metadata=os.fstat(marker_fd)
        marker_payload=os.read(marker_fd,256); marker_after=os.fstat(marker_fd); os.close(marker_fd)
    except BaseException: _deny()
    if (marker_metadata.st_dev,marker_metadata.st_ino,marker_metadata.st_size,marker_metadata.st_mtime_ns,marker_metadata.st_ctime_ns)!=(marker_after.st_dev,marker_after.st_ino,marker_after.st_size,marker_after.st_mtime_ns,marker_after.st_ctime_ns) or not token_sink or '\x00' in token_sink or len(token_sink.encode('utf-8'))>4096 or not os.path.isabs(token_sink) or os.path.abspath(token_sink)!=token_sink or os.path.basename(token_sink)!='unavailable' or os.path.basename(token_marker)!='.ephemeral-token-persistence-disabled' or not stat.S_ISREG(marker_metadata.st_mode) or marker_metadata.st_uid!=os.geteuid() or marker_metadata.st_nlink!=1 or stat.S_IMODE(marker_metadata.st_mode)!=0o400 or marker_payload!=b'trusted controller keeps the Quack attach credential in memory\n': _deny()
    capsule_fd=int(sys.argv.pop(1)); pin_text=sys.argv.pop(1); admission_text=sys.argv.pop(1)
    native_fd=int(sys.argv.pop(1)); native_text=sys.argv.pop(1); module=sys.argv.pop(1)
    expected_bootstrap=sys.argv.pop(1); expected_python=sys.argv.pop(1)
    if capsule_fd<3 or native_fd<3 or capsule_fd==native_fd: _deny()
    if module not in {'ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler','ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor','ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon'}: _deny()
    command_line=open('/proc/self/cmdline','rb').read().split(b'\0')
    if command_line[1:5]!=[b'-I',b'-S',b'-B',b'-c']: _deny()
    code_index=command_line.index(b'-c')+1
    if _sha(command_line[code_index])!=expected_bootstrap: _deny()
    executable=os.open('/proc/self/exe',os.O_RDONLY|getattr(os,'O_CLOEXEC',0))
    try:
        executable_hash=hashlib.sha256()
        while True:
            block=os.read(executable,65536)
            if not block: break
            executable_hash.update(block)
    finally: os.close(executable)
    if 'sha256:'+executable_hash.hexdigest()!=expected_python: _deny()
    pin_payload=_decode(pin_text); admission=_decode(admission_text); native_payload=_decode(native_text)
    archive_sha=pin_payload.get('archive_sha256')
    if type(archive_sha) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}',archive_sha) is None: _deny()
    required=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SEAL
    metadata=os.fstat(capsule_fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size<=0 or fcntl.fcntl(capsule_fd,fcntl.F_GET_SEALS)&required!=required: _deny()
    archive_hash=hashlib.sha256(); offset=0
    while offset<metadata.st_size:
        block=os.pread(capsule_fd,min(65536,metadata.st_size-offset),offset)
        if not block: break
        archive_hash.update(block); offset+=len(block)
    if offset!=metadata.st_size or 'sha256:'+archive_hash.hexdigest()!=archive_sha: _deny()
    archive='/proc/self/fd/'+str(capsule_fd); path_metadata=os.stat(archive)
    if (path_metadata.st_dev,path_metadata.st_ino)!=(metadata.st_dev,metadata.st_ino): _deny()
    stdlib=[]
    for entry in sys.path:
        if type(entry) is not str or not entry or not os.path.isabs(entry): continue
        lowered=entry.casefold()
        if 'site-packages' in lowered or 'dist-packages' in lowered: continue
        stdlib.append(entry)
    sys.path[:]=[archive,*dict.fromkeys(stdlib)]
    import importlib,importlib.machinery,runpy
    allowed_meta=(importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder)
    if any(finder not in allowed_meta for finder in sys.meta_path): _deny()
    import ipfs_accelerate_py as accepted_root
    prefix=archive+'/'
    root_origin=getattr(accepted_root,'__file__',None)
    if type(root_origin) is not str or not root_origin.startswith(prefix): _deny()
    from ipfs_accelerate_py import agent_implementation_route as route
    pin=route.parse_lgcvf_configured_board_live_capsule_pin(pin_payload)
    if pin.to_json()!=pin_text or route.verify_lgcvf_configured_board_live_sealed_capsule(pin,capsule_fd)!=archive: _deny()
    if tuple(pin.python_path_prefixes)!=('.', 'ipfs_datasets_py'): _deny()
    capsule_roots=[]
    for root in pin.python_path_prefixes:
        if type(root) is not str or not root or root.startswith(('/', '\\')) or '\x00' in root or '..' in root.replace('\\','/').split('/'): _deny()
        projected=archive if root=='.' else archive+'/'+root.strip('/')
        if projected not in capsule_roots: capsule_roots.append(projected)
    from ipfs_accelerate_py import llm_router as router
    native=router.parse_agent_supervisor_native_dependency_launch(native_payload)
    if native.to_json()!=native_text or native.descriptor.descriptor!=native_fd: _deny()
    if router.verify_agent_supervisor_native_dependency_sealed_fd(native)!='/proc/self/fd/'+str(native_fd): _deny()
    if pin.native_authorization_id!=native.accepted_authorization_id or pin.native_dependency_id!=native.pin.dependency_id: _deny()
    expected_fields={'schema','admission_id','capsule_id','capsule_archive_sha256','source_head','source_tree','candidate_config_path','candidate_config_sha256','board_namespace','scheduler_schema','validator_sha256','materializer_sha256','operator_sha256','native_authorization_id','native_dependency_id','task_source_kind','authority_mode','state_schema_revision','lane_names'}
    if set(admission)!=expected_fields: _deny()
    unsigned=dict(admission); unsigned.pop('admission_id',None)
    admission_id=_sha(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8'))
    if admission.get('schema')!='ipfs_accelerate_py.agent_supervisor.lgcvf-configured-board-live-admission@1' or admission.get('admission_id')!=admission_id: _deny()
    expected={'capsule_id':pin.capsule_id,'capsule_archive_sha256':pin.archive_sha256,'source_head':pin.source_head,'source_tree':pin.source_tree,'candidate_config_path':pin.candidate_config_path,'candidate_config_sha256':pin.candidate_config_sha256,'board_namespace':'logic-governed-compositional-verification-fabric-v1','scheduler_schema':'ipfs_accelerate_py.agent_supervisor.logic_governed_compositional_verification_fabric.scheduler_config@1','validator_sha256':pin.validator_sha256,'materializer_sha256':pin.materializer_sha256,'operator_sha256':pin.operator_sha256,'native_authorization_id':pin.native_authorization_id,'native_dependency_id':pin.native_dependency_id,'task_source_kind':'duckdb','authority_mode':'quack','state_schema_revision':'datasets-authoritative-operational-v1','lane_names':['lgcvf-quack-lane-0','lgcvf-quack-lane-1','lgcvf-quack-lane-2','lgcvf-quack-lane-3']}
    if any(admission.get(name)!=value for name,value in expected.items()): _deny()
    python_identity=pin.python_identity
    accepted_python=_field(python_identity,'python_executable_sha256')
    if accepted_python is None: accepted_python=_field(python_identity,'executable_sha256')
    if accepted_python!=expected_python or native.pin.python_executable_sha256!=expected_python: _deny()
    router.preload_agent_supervisor_native_dependency(native)
    dataset_prefix=archive+'/ipfs_datasets_py/'
    projected_roots=list(reversed(capsule_roots))
    sys.path[:]=[*projected_roots,*[entry for entry in sys.path if entry not in capsule_roots]]
    package_name='ipfs_accelerate_py.agent_supervisor'
    if any(name==package_name or name.startswith(package_name+'.') for name in sys.modules): _deny()
    package=importlib.import_module(package_name)
    package_origin=getattr(package,'__file__',None)
    if type(package_origin) is not str or not package_origin.startswith(prefix): _deny()
    setattr(accepted_root,'agent_supervisor',package)
    if module in sys.modules: _deny()
    namespace=runpy.run_module(module,run_name=module,alter_sys=True)
    if module in sys.modules: _deny()
    target_origin=namespace.get('__file__')
    if type(target_origin) is not str or not target_origin.startswith(prefix): _deny()
    for name,loaded in tuple(sys.modules.items()):
        if name=='ipfs_accelerate_py' or name.startswith('ipfs_accelerate_py.'):
            origin=getattr(loaded,'__file__',None)
            if type(origin) is not str or not origin.startswith(prefix): _deny()
        if name=='ipfs_datasets_py' or name.startswith('ipfs_datasets_py.'):
            origin=getattr(loaded,'__file__',None)
            if type(origin) is not str or not origin.startswith(dataset_prefix): _deny()
    main=namespace.get('main')
    if not callable(main): _deny()
    raise SystemExit(main())
except SystemExit: raise
except BaseException: raise SystemExit(78)
'''
LGCVF_CONFIGURED_BOARD_LIVE_BOOTSTRAP_SHA256 = (
    "sha256:"
    + hashlib.sha256(
        LGCVF_CONFIGURED_BOARD_LIVE_BOOTSTRAP.encode("utf-8")
    ).hexdigest()
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


def _strict_canonical_json_object(
    value: str,
    *,
    label: str,
    maximum_bytes: int = 262_144,
) -> dict[str, object]:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum_bytes
    ):
        raise ValueError(f"{label} JSON is invalid")
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} JSON is invalid") from exc
    if type(payload) is not dict:
        raise ValueError(f"{label} JSON must be an exact object")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if canonical != value:
        raise ValueError(f"{label} JSON is not canonical")
    return payload


@dataclass(frozen=True, slots=True)
class LgcvfConfiguredBoardLiveAdmission:
    """Closed authorization joining the LGCVF capsule and native launch."""

    schema: str
    admission_id: str
    capsule_id: str
    capsule_archive_sha256: str
    source_head: str
    source_tree: str
    candidate_config_path: str
    candidate_config_sha256: str
    board_namespace: str
    scheduler_schema: str
    validator_sha256: str
    materializer_sha256: str
    operator_sha256: str
    native_authorization_id: str
    native_dependency_id: str
    task_source_kind: str
    authority_mode: str
    state_schema_revision: str
    lane_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "admission_id": self.admission_id,
            "capsule_id": self.capsule_id,
            "capsule_archive_sha256": self.capsule_archive_sha256,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "candidate_config_path": self.candidate_config_path,
            "candidate_config_sha256": self.candidate_config_sha256,
            "board_namespace": self.board_namespace,
            "scheduler_schema": self.scheduler_schema,
            "validator_sha256": self.validator_sha256,
            "materializer_sha256": self.materializer_sha256,
            "operator_sha256": self.operator_sha256,
            "native_authorization_id": self.native_authorization_id,
            "native_dependency_id": self.native_dependency_id,
            "task_source_kind": self.task_source_kind,
            "authority_mode": self.authority_mode,
            "state_schema_revision": self.state_schema_revision,
            "lane_names": list(self.lane_names),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


def _lgcvf_live_admission_id(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("admission_id", None)
    return "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def parse_lgcvf_configured_board_live_admission(
    value: str | Mapping[str, object],
) -> LgcvfConfiguredBoardLiveAdmission:
    """Strictly decode the one closed LGCVF live launch admission."""

    if isinstance(value, str):
        payload = _strict_canonical_json_object(
            value,
            label="LGCVF configured-board live admission",
        )
    elif type(value) is dict:
        payload = dict(value)
    else:
        raise ValueError(
            "LGCVF configured-board live admission must be an exact object"
        )
    expected = {
        "schema",
        "admission_id",
        "capsule_id",
        "capsule_archive_sha256",
        "source_head",
        "source_tree",
        "candidate_config_path",
        "candidate_config_sha256",
        "board_namespace",
        "scheduler_schema",
        "validator_sha256",
        "materializer_sha256",
        "operator_sha256",
        "native_authorization_id",
        "native_dependency_id",
        "task_source_kind",
        "authority_mode",
        "state_schema_revision",
        "lane_names",
    }
    lane_names = payload.get("lane_names")
    if (
        set(payload) != expected
        or any(
            type(payload.get(name)) is not str or not payload.get(name)
            for name in expected - {"lane_names"}
        )
        or type(lane_names) is not list
        or any(type(name) is not str or not name for name in lane_names)
    ):
        raise ValueError(
            "LGCVF configured-board live admission fields are not exact"
        )
    admission = LgcvfConfiguredBoardLiveAdmission(
        **{
            **{name: str(payload[name]) for name in expected - {"lane_names"}},
            "lane_names": tuple(lane_names),
        }
    )
    sha256_pattern = r"sha256:[0-9a-f]{64}"
    if (
        admission.schema != LGCVF_CONFIGURED_BOARD_LIVE_ADMISSION_SCHEMA
        or admission.admission_id != _lgcvf_live_admission_id(payload)
        or re.fullmatch(sha256_pattern, admission.admission_id) is None
        or re.fullmatch(sha256_pattern, admission.capsule_id) is None
        or re.fullmatch(sha256_pattern, admission.capsule_archive_sha256)
        is None
        or re.fullmatch(sha256_pattern, admission.candidate_config_sha256)
        is None
        or re.fullmatch(sha256_pattern, admission.validator_sha256) is None
        or re.fullmatch(sha256_pattern, admission.materializer_sha256) is None
        or re.fullmatch(sha256_pattern, admission.operator_sha256) is None
        or re.fullmatch(sha256_pattern, admission.native_authorization_id)
        is None
        or re.fullmatch(sha256_pattern, admission.native_dependency_id)
        is None
        or re.fullmatch(r"[0-9a-f]{40}", admission.source_head) is None
        or re.fullmatch(r"[0-9a-f]{40}", admission.source_tree) is None
        or admission.candidate_config_path
        != LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH
        or admission.board_namespace != LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE
        or admission.scheduler_schema
        != LGCVF_CONFIGURED_BOARD_LIVE_SCHEDULER_SCHEMA
        or admission.task_source_kind != TASK_SOURCE_DUCKDB
        or admission.authority_mode != AUTHORITY_MODE_QUACK
        or admission.state_schema_revision
        != DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION
        or admission.lane_names != LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES
    ):
        raise ValueError(
            "LGCVF configured-board live admission identity is invalid"
        )
    if admission.to_json() != json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ):
        raise ValueError(
            "LGCVF configured-board live admission changed during decode"
        )
    return admission


def build_lgcvf_configured_board_live_admission(
    capsule_pin: object,
    native_launch: object,
) -> LgcvfConfiguredBoardLiveAdmission:
    """Build the sole accepted LGCVF profile from already verified DTOs."""

    pin_fields = {
        name: getattr(capsule_pin, name, None)
        for name in (
            "capsule_id",
            "archive_sha256",
            "source_head",
            "source_tree",
            "candidate_config_path",
            "candidate_config_sha256",
            "validator_sha256",
            "materializer_sha256",
            "operator_sha256",
            "native_authorization_id",
            "native_dependency_id",
        )
    }
    native_pin = getattr(native_launch, "pin", None)
    if (
        any(type(value) is not str or not value for value in pin_fields.values())
        or getattr(native_launch, "accepted_authorization_id", None)
        != pin_fields["native_authorization_id"]
        or getattr(native_pin, "dependency_id", None)
        != pin_fields["native_dependency_id"]
    ):
        raise ValueError(
            "LGCVF capsule and native launch authority are not identical"
        )
    payload: dict[str, object] = {
        "schema": LGCVF_CONFIGURED_BOARD_LIVE_ADMISSION_SCHEMA,
        "admission_id": "",
        "capsule_id": pin_fields["capsule_id"],
        "capsule_archive_sha256": pin_fields["archive_sha256"],
        "source_head": pin_fields["source_head"],
        "source_tree": pin_fields["source_tree"],
        "candidate_config_path": pin_fields["candidate_config_path"],
        "candidate_config_sha256": pin_fields["candidate_config_sha256"],
        "board_namespace": LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,
        "scheduler_schema": LGCVF_CONFIGURED_BOARD_LIVE_SCHEDULER_SCHEMA,
        "validator_sha256": pin_fields["validator_sha256"],
        "materializer_sha256": pin_fields["materializer_sha256"],
        "operator_sha256": pin_fields["operator_sha256"],
        "native_authorization_id": pin_fields["native_authorization_id"],
        "native_dependency_id": pin_fields["native_dependency_id"],
        "task_source_kind": TASK_SOURCE_DUCKDB,
        "authority_mode": AUTHORITY_MODE_QUACK,
        "state_schema_revision": (
            DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION
        ),
        "lane_names": list(LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES),
    }
    payload["admission_id"] = _lgcvf_live_admission_id(payload)
    return parse_lgcvf_configured_board_live_admission(payload)


@dataclass(frozen=True, slots=True)
class LgcvfConfiguredBoardLiveContext:
    capsule_pin: object
    capsule_descriptor: int
    admission: LgcvfConfiguredBoardLiveAdmission
    native_launch: object
    native_descriptor: int
    capsule_pin_json: str
    admission_json: str
    native_launch_json: str

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return tuple(
            sorted({self.capsule_descriptor, self.native_descriptor})
        )


def verify_lgcvf_configured_board_live_context(
    *,
    capsule_pin_json: str,
    capsule_descriptor: int,
    admission_json: str,
    native_launch_json: str,
    native_descriptor: int,
) -> LgcvfConfiguredBoardLiveContext:
    """Revalidate one capsule/native/admission join without importing DuckDB."""

    if (
        isinstance(capsule_descriptor, bool)
        or not isinstance(capsule_descriptor, int)
        or isinstance(native_descriptor, bool)
        or not isinstance(native_descriptor, int)
        or capsule_descriptor < 3
        or native_descriptor < 3
        or capsule_descriptor == native_descriptor
    ):
        raise ValueError("LGCVF configured-board live descriptors are invalid")
    from ipfs_accelerate_py.agent_implementation_route import (
        parse_lgcvf_configured_board_live_capsule_pin,
        verify_lgcvf_configured_board_live_sealed_capsule,
    )
    from ipfs_accelerate_py.llm_router import (
        parse_agent_supervisor_native_dependency_launch,
        verify_agent_supervisor_native_dependency_sealed_fd,
    )

    capsule_payload = _strict_canonical_json_object(
        capsule_pin_json,
        label="LGCVF configured-board live capsule pin",
    )
    capsule_pin = parse_lgcvf_configured_board_live_capsule_pin(
        capsule_payload
    )
    if getattr(capsule_pin, "to_json", lambda: "")() != capsule_pin_json:
        raise ValueError("LGCVF configured-board live capsule pin is not canonical")
    capsule_path = verify_lgcvf_configured_board_live_sealed_capsule(
        capsule_pin,
        capsule_descriptor,
    )
    native_payload = _strict_canonical_json_object(
        native_launch_json,
        label="LGCVF configured-board live native launch",
        maximum_bytes=32_768,
    )
    native_launch = parse_agent_supervisor_native_dependency_launch(
        native_payload
    )
    if getattr(native_launch, "to_json", lambda: "")() != native_launch_json:
        raise ValueError("LGCVF configured-board native launch is not canonical")
    if getattr(getattr(native_launch, "descriptor", None), "descriptor", -1) != (
        native_descriptor
    ):
        raise ValueError("LGCVF configured-board native descriptor was substituted")
    native_path = verify_agent_supervisor_native_dependency_sealed_fd(
        native_launch
    )
    admission = parse_lgcvf_configured_board_live_admission(admission_json)
    expected_admission = build_lgcvf_configured_board_live_admission(
        capsule_pin,
        native_launch,
    )
    if admission != expected_admission:
        raise ValueError("LGCVF configured-board live admission is foreign")
    if (
        capsule_path != f"/proc/self/fd/{capsule_descriptor}"
        or native_path != f"/proc/self/fd/{native_descriptor}"
    ):
        raise ValueError("LGCVF configured-board live descriptor path drifted")
    return LgcvfConfiguredBoardLiveContext(
        capsule_pin=capsule_pin,
        capsule_descriptor=capsule_descriptor,
        admission=admission,
        native_launch=native_launch,
        native_descriptor=native_descriptor,
        capsule_pin_json=capsule_pin_json,
        admission_json=admission_json,
        native_launch_json=native_launch_json,
    )


def _python_executable_sha256(python_executable: str) -> tuple[str, str]:
    executable = Path(python_executable).resolve(strict=True)
    metadata = os.stat(executable, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("sealed control-plane Python executable is not regular")
    digest = hashlib.sha256()
    descriptor = os.open(
        executable,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(item):
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
    if identity(before) != identity(after) or identity(before) != identity(metadata):
        raise ValueError("sealed control-plane Python executable changed")
    return str(executable), "sha256:" + digest.hexdigest()


def build_sealed_control_plane_module_command(
    *,
    python_executable: str,
    pin: AgentImplementationControlPlanePin,
    descriptor: int,
    module_name: str,
    argv: Sequence[str],
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
    executable, executable_sha256 = _python_executable_sha256(python_executable)
    return [
        executable,
        "-I",
        "-c",
        SEALED_CONTROL_PLANE_BOOTSTRAP,
        str(descriptor),
        accepted_control_plane_pin_json(pin),
        module_name,
        SEALED_CONTROL_PLANE_BOOTSTRAP_SHA256,
        executable_sha256,
        *[str(item) for item in argv],
    ]


def build_lgcvf_configured_board_live_module_command(
    *,
    python_executable: str,
    capsule_pin_json: str,
    capsule_descriptor: int,
    admission_json: str,
    native_launch_json: str,
    native_descriptor: int,
    module_name: str,
    argv: Sequence[str],
) -> list[str]:
    """Build an isolated LGCVF role launch from two independently sealed FDs."""

    if module_name not in LGCVF_CONFIGURED_BOARD_LIVE_MODULES:
        raise ValueError("LGCVF configured-board live target module is not allowed")
    context = verify_lgcvf_configured_board_live_context(
        capsule_pin_json=capsule_pin_json,
        capsule_descriptor=capsule_descriptor,
        admission_json=admission_json,
        native_launch_json=native_launch_json,
        native_descriptor=native_descriptor,
    )
    executable, executable_sha256 = _python_executable_sha256(
        python_executable
    )
    python_identity = getattr(context.capsule_pin, "python_identity", None)
    accepted_python_sha256 = getattr(
        python_identity,
        "python_executable_sha256",
        getattr(python_identity, "executable_sha256", ""),
    )
    native_python_sha256 = getattr(
        getattr(context.native_launch, "pin", None),
        "python_executable_sha256",
        "",
    )
    if (
        executable_sha256 != accepted_python_sha256
        or executable_sha256 != native_python_sha256
    ):
        raise ValueError(
            "LGCVF configured-board Python executable differs from admission"
        )
    return [
        executable,
        "-I",
        "-S",
        "-B",
        "-c",
        LGCVF_CONFIGURED_BOARD_LIVE_BOOTSTRAP,
        str(context.capsule_descriptor),
        context.capsule_pin_json,
        context.admission_json,
        str(context.native_descriptor),
        context.native_launch_json,
        module_name,
        LGCVF_CONFIGURED_BOARD_LIVE_BOOTSTRAP_SHA256,
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
QUACK_TOKEN_FILE_ENV = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE"
BOARD_EXTENSION_INSTALL_POLICY_ENV = (
    "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
)
BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY = "load_only"
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

_PLAN_BOUND_PROFILE_ENV_NAMES = frozenset(
    {
        *ORDERED_IMPLEMENTATION_PROVIDER_ROUTE,
        *_ROUTE_AUTHORIZATION_ENV_NAMES,
        *_PROVIDER_EXECUTABLE_ENV_NAMES,
        *DATABASE_PROGRAM_ENV_NAMES,
        PROVIDER_EXTERNAL_ISOLATION_ENV,
        TRUSTED_DUCKDB_HOME_ENV,
        QUACK_TOKEN_FILE_ENV,
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
    }
    projected = {
        name: str(value)
        for name, value in environment.items()
        if name in allowed_names
    }
    trusted_home = str(projected.get(TRUSTED_DUCKDB_HOME_ENV, "") or "")
    if trusted_home:
        repository_root = str(environment.get(REPOSITORY_ROOT_ENV, "") or "")
        projected.update(
            _trusted_duckdb_runtime_environment(
                environment,
                repository_root=Path(repository_root),
            )
        )
    else:
        projected.pop(TRUSTED_PYTHON_USER_BASE_ENV, None)
        for name in TRUSTED_RUNTIME_CACHE_ENV_NAMES:
            projected.pop(name, None)
    projected["PATH"] = "/usr/bin:/bin"
    return projected


def _lgcvf_configured_board_live_positive_child_environment(
    environment: Mapping[str, str],
    *,
    common_args: Sequence[str],
) -> dict[str, str]:
    """Project only the admitted LGCVF state credential and control bindings."""

    allowed_names = {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        *_PLAN_BOUND_LIFECYCLE_ENV_NAMES,
        *_PLAN_BOUND_PROFILE_ENV_NAMES,
    }
    projected = {
        name: str(value)
        for name, value in environment.items()
        if name in allowed_names
    }
    trusted_home = str(
        projected.get(TRUSTED_DUCKDB_HOME_ENV, "") or ""
    )
    if trusted_home:
        repository_root = str(environment.get(REPOSITORY_ROOT_ENV, "") or "")
        home = _validate_trusted_duckdb_home(
            trusted_home,
            repository_root=repository_root,
            observed_home=str(environment.get("HOME", "") or ""),
        )
        cache_root = home / ".cache"
        xdg_cache = cache_root / "xdg"
        cuda_cache = cache_root / "cuda"
        for directory in (cache_root, xdg_cache, cuda_cache):
            observed = os.lstat(directory)
            if (
                directory.resolve(strict=True) != directory
                or not stat.S_ISDIR(observed.st_mode)
                or stat.S_ISLNK(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o700
            ):
                raise ValueError("LGCVF trusted runtime cache is unsafe")
        projected.update(
            {
                "HOME": str(home),
                TRUSTED_DUCKDB_HOME_ENV: str(home),
                TRUSTED_XDG_CACHE_HOME_ENV: str(xdg_cache),
                TRUSTED_CUDA_CACHE_PATH_ENV: str(cuda_cache),
                TRUSTED_CUDA_CACHE_DISABLE_ENV: "1",
            }
        )
    handles = _profile_option_values(common_args, "--endpoint-secret-handle")
    if handles != ("env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",):
        raise ValueError("LGCVF live Quack credential handle is not exact")
    credential_name = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
    from .process_security import state_authority_credential

    credential = state_authority_credential(
        credential_name,
        environment=environment,
    )
    if not credential:
        raise ValueError("LGCVF live Quack credential is unavailable")
    projected[credential_name] = credential
    token_sink_value = str(environment.get(QUACK_TOKEN_FILE_ENV, "") or "")
    if (
        not token_sink_value
        or "\x00" in token_sink_value
        or len(token_sink_value.encode("utf-8")) > 4096
    ):
        raise ValueError("LGCVF live Quack token sink is unavailable")
    token_sink = Path(token_sink_value)
    token_sink_marker = token_sink.parent
    marker_descriptor = -1
    try:
        marker_descriptor = os.open(
            token_sink_marker,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        marker_metadata = os.fstat(marker_descriptor)
        marker_payload = os.read(marker_descriptor, 256)
        marker_after = os.fstat(marker_descriptor)
    except OSError as exc:
        raise ValueError("LGCVF live Quack token sink is unavailable") from exc
    finally:
        if marker_descriptor >= 0:
            os.close(marker_descriptor)
    if (
        (
            marker_metadata.st_dev,
            marker_metadata.st_ino,
            marker_metadata.st_size,
            marker_metadata.st_mtime_ns,
            marker_metadata.st_ctime_ns,
        )
        != (
            marker_after.st_dev,
            marker_after.st_ino,
            marker_after.st_size,
            marker_after.st_mtime_ns,
            marker_after.st_ctime_ns,
        )
        or
        not token_sink.is_absolute()
        or Path(os.path.abspath(token_sink_value)) != token_sink
        or token_sink.name != "unavailable"
        or token_sink_marker.name != ".ephemeral-token-persistence-disabled"
        or not stat.S_ISREG(marker_metadata.st_mode)
        or marker_metadata.st_uid != os.geteuid()
        or marker_metadata.st_nlink != 1
        or stat.S_IMODE(marker_metadata.st_mode) != 0o400
        or marker_payload
        != b"trusted controller keeps the Quack attach credential in memory\n"
    ):
        raise ValueError("LGCVF live Quack token sink is unsafe")
    projected[QUACK_TOKEN_FILE_ENV] = token_sink_value
    extension_policy = str(
        environment.get(BOARD_EXTENSION_INSTALL_POLICY_ENV, "") or ""
    )
    if extension_policy != BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY:
        raise ValueError("LGCVF live DuckDB extension policy is not load-only")
    projected[BOARD_EXTENSION_INSTALL_POLICY_ENV] = extension_policy
    for name in tuple(projected):
        if (
            name.startswith(("LD_", "PYTHON"))
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


def _lgcvf_configured_board_live_profile_environment(
    environment: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    """Bind the non-secret token sink in addition to the closed lane profile."""

    names = {
        *_PLAN_BOUND_PROFILE_ENV_NAMES,
        QUACK_TOKEN_FILE_ENV,
        BOARD_EXTENSION_INSTALL_POLICY_ENV,
    }
    return tuple(
        sorted(
            (name, str(environment[name]))
            for name in names
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
    if not trusted_home or not python_user_base:
        raise ValueError("trusted DuckDB HOME and Python user base must be paired")
    home = _validate_trusted_duckdb_home(
        trusted_home,
        repository_root=str(repository_root.resolve()),
        observed_home=str(environment.get("HOME", "") or ""),
    )
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
    return {
        "HOME": str(home),
        TRUSTED_DUCKDB_HOME_ENV: str(home),
        TRUSTED_PYTHON_USER_BASE_ENV: str(user_base),
        TRUSTED_XDG_CACHE_HOME_ENV: str(xdg_cache),
        TRUSTED_CUDA_CACHE_PATH_ENV: str(cuda_cache),
        TRUSTED_CUDA_CACHE_DISABLE_ENV: "1",
        TRUSTED_PYTHONDONTWRITEBYTECODE_ENV: "1",
    }


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
    claim_policy: Mapping[str, Any] | None = None

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

        claim_policy = dict(self.claim_policy or {})
        if claim_policy:
            expected_fields = {
                "schema",
                "task_prefix",
                "task_shard_count",
                "strict_task_sharding",
                "idle_lane_work_stealing",
            }
            shard_count = claim_policy.get("task_shard_count")
            if (
                set(claim_policy) != expected_fields
                or claim_policy.get("schema")
                != "ipfs_accelerate_py/agent-supervisor/database-claim-policy@1"
                or not str(claim_policy.get("task_prefix") or "").strip()
                or isinstance(shard_count, bool)
                or not isinstance(shard_count, int)
                or shard_count <= 1
                or claim_policy.get("strict_task_sharding") is not True
                or claim_policy.get("idle_lane_work_stealing") != "virgin-transfer"
            ):
                raise DatabaseProgramConfigError(
                    "claim_policy must be an exact strict virgin-transfer policy"
                )
            claim_policy["task_prefix"] = str(
                claim_policy["task_prefix"]
            ).strip()
        object.__setattr__(
            self,
            "claim_policy",
            MappingProxyType(claim_policy),
        )

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
        payload = {
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
        if self.claim_policy:
            payload["claim_policy"] = dict(self.claim_policy)
        return payload

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
        if "claim_policy" in payload and not isinstance(
            payload.get("claim_policy"), Mapping
        ):
            raise DatabaseProgramConfigError("claim_policy must be an object")
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
            claim_policy=(
                payload.get("claim_policy")
                if isinstance(payload.get("claim_policy"), Mapping)
                else None
            ),
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
    # Provider children also must not receive the supervisor's state-authority
    # bindings; they operate on worktree files only.
    for name in DATABASE_PROGRAM_ENV_NAMES:
        cleaned.pop(name, None)
    cleaned.pop(QUACK_TOKEN_FILE_ENV, None)
    cleaned.pop(BOARD_EXTENSION_INSTALL_POLICY_ENV, None)
    cleaned.pop(REPOSITORY_ROOT_ENV, None)
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


def _lgcvf_configured_board_live_embedded_config(
    context: LgcvfConfiguredBoardLiveContext,
) -> dict[str, object]:
    """Read the admitted candidate config only from the sealed capsule FD."""

    pin = context.capsule_pin
    member_name = str(getattr(pin, "candidate_config_path", "") or "")
    if member_name != LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH:
        raise ValueError("LGCVF configured-board capsule config path is foreign")
    try:
        with zipfile.ZipFile(
            f"/proc/self/fd/{context.capsule_descriptor}",
            mode="r",
            allowZip64=False,
        ) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)) or member_name not in names:
                raise ValueError("LGCVF capsule config member is not unique")
            info = archive.getinfo(member_name)
            mode = info.external_attr >> 16
            if (
                info.is_dir()
                or stat.S_ISLNK(mode)
                or info.file_size <= 0
                or info.file_size > 2_097_152
                or info.compress_size > 2_097_152
            ):
                raise ValueError("LGCVF capsule config member is invalid")
            raw = archive.read(info)
    except (KeyError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("LGCVF capsule config member is unavailable") from exc
    if (
        len(raw) != info.file_size
        or "sha256:" + hashlib.sha256(raw).hexdigest()
        != getattr(pin, "candidate_config_sha256", "")
    ):
        raise ValueError("LGCVF capsule config bytes drifted")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("LGCVF capsule config is invalid JSON") from exc
    if type(payload) is not dict:
        raise ValueError("LGCVF capsule config must be an exact object")
    return payload


def _lgcvf_live_exact_option(
    argv: Sequence[str],
    option: str,
    expected: str,
) -> None:
    if _profile_option_values(argv, option) != (str(expected),):
        raise ValueError(
            f"LGCVF configured-board live {option} differs from the capsule"
        )


def _verify_lgcvf_configured_board_live_profile(
    *,
    tracks: Sequence[SupervisorTrack],
    repo_root: Path,
    common_args: Sequence[str],
    context: LgcvfConfiguredBoardLiveContext,
    require_complete_lane_set: bool = True,
) -> tuple[SupervisorTrack, ...]:
    """Bind the runner's four projected lanes to the sealed config bytes."""

    root = _canonical_accepted_tree_root(Path(repo_root))
    payload = _lgcvf_configured_board_live_embedded_config(context)
    database = payload.get("database_program")
    runtime_paths = payload.get("runtime_paths")
    lanes = payload.get("lanes")
    if (
        payload.get("schema") != LGCVF_CONFIGURED_BOARD_LIVE_SCHEDULER_SCHEMA
        or payload.get("board_namespace")
        != LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE
        or payload.get("max_lanes") != 4
        or payload.get("strict_task_sharding") is not True
        or payload.get("idle_lane_work_stealing") != "virgin-transfer"
        or type(database) is not dict
        or type(runtime_paths) is not dict
        or type(lanes) is not list
        or len(lanes) != 4
    ):
        raise ValueError("LGCVF capsule config is not the closed live profile")
    expected_lane_records = [
        {
            "index": index,
            "name": LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES[index],
            "strict_shard_remainder": index,
        }
        for index in range(4)
    ]
    for observed, expected in zip(lanes, expected_lane_records):  # noqa: B905
        if type(observed) is not dict or any(
            observed.get(name) != value for name, value in expected.items()
        ):
            raise ValueError("LGCVF capsule lane topology is foreign")
    common = tuple(str(item) for item in common_args)
    exact_options = {
        "--board-namespace": LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,
        "--task-prefix": str(payload.get("task_prefix") or ""),
        "--merge-target-branch": str(payload.get("merge_target_branch") or ""),
        "--idle-lane-work-stealing": str(
            payload.get("idle_lane_work_stealing") or ""
        ),
        "--task-source-kind": str(database.get("task_source_kind") or ""),
        "--authority-mode": str(database.get("authority_mode") or ""),
        "--state-failover-policy": str(database.get("failover_policy") or ""),
        "--endpoint-secret-handle": str(
            database.get("endpoint_secret_handle") or ""
        ),
        "--quack-endpoint": str(database.get("quack_endpoint") or ""),
        "--state-store-id": str(database.get("store_id") or ""),
        "--state-store-generation": str(
            database.get("store_generation") or ""
        ),
        "--state-schema-revision": str(
            database.get("schema_revision") or ""
        ),
        "--event-store-path": str(database.get("event_store_path") or ""),
        "--runtime-registry-path": str(
            database.get("runtime_registry_path") or ""
        ),
        "--export-profile": str(database.get("export_profile") or ""),
        "--worktree-root": str(
            root / str(runtime_paths.get("worktrees") or "")
        ),
        "--merge-queue-dir": str(
            root / str(runtime_paths.get("merge_queue") or "")
        ),
        "--todo-path": str(root / str(payload.get("taskboard_path") or "")),
    }
    if (
        database.get("task_source_kind") != TASK_SOURCE_DUCKDB
        or database.get("authority_mode") != AUTHORITY_MODE_QUACK
        or database.get("schema_revision")
        != DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION
        or database.get("failover_policy") != FAILOVER_FAIL_CLOSED
        or database.get("explicit_legacy") is not False
        or database.get("authoritative_transactional_data_model") is not True
        or database.get("authoritative_records_schema_cas_and_fencing")
        is not True
        or database.get("endpoint_secret_handle")
        != "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
        or common.count("--strict-task-sharding") != 1
    ):
        raise ValueError("LGCVF live database profile is not exact Quack authority")
    for option, expected in exact_options.items():
        if not expected:
            raise ValueError(f"LGCVF capsule config lacks {option}")
        _lgcvf_live_exact_option(common, option, expected)
    if require_complete_lane_set:
        if tuple(track.name for track in tracks) != (
            LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES
        ):
            raise ValueError("LGCVF live runner lane names are not exact")
        indexed_tracks = tuple(enumerate(tracks))
    else:
        if (
            len(tracks) != 1
            or tracks[0].name not in LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES
        ):
            raise ValueError("LGCVF live runner lane identity is not exact")
        indexed_tracks = (
            (
                LGCVF_CONFIGURED_BOARD_LIVE_LANE_NAMES.index(tracks[0].name),
                tracks[0],
            ),
        )
    entry = root / PLAN_BOUND_ACCEPTED_ENTRY_PATH
    _lexical_contained_path(root, entry, require_regular=True)
    state_root = root / str(runtime_paths.get("state") or "")
    resolved_tracks: list[SupervisorTrack] = []
    for index, track in indexed_tracks:
        resolved = track.resolve(root)
        if (
            resolved.module_name
            or resolved.script_path != entry
            or resolved.log_path.parent != state_root / f"lane-{index}"
            or resolved.supervisor_pid_path.parent
            != state_root / f"lane-{index}"
            or resolved.daemon_pid_path.parent != state_root / f"lane-{index}"
            or (
                resolved.supervisor_status_path is not None
                and resolved.supervisor_status_path.parent
                != state_root / f"lane-{index}"
            )
            or not _path_within(resolved.log_path, root)
            or not _path_within(resolved.supervisor_pid_path, root)
            or not _path_within(resolved.daemon_pid_path, root)
        ):
            raise ValueError("LGCVF live lane paths are not exact")
        _lgcvf_live_exact_option(
            resolved.extra_args,
            "--state-dir",
            str(state_root / f"lane-{index}"),
        )
        _lgcvf_live_exact_option(
            resolved.extra_args,
            "--state-prefix",
            f"lgcvf_lane_{index}",
        )
        _lgcvf_live_exact_option(
            resolved.extra_args,
            "--task-shard-count",
            "4",
        )
        _lgcvf_live_exact_option(
            resolved.extra_args,
            "--task-shard-index",
            str(index),
        )
        resolved_tracks.append(resolved)
    return tuple(resolved_tracks)


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


class LgcvfConfiguredBoardLiveProcessBirthError(RuntimeError):
    """A live capsule gate was fenced before its typed birth was captured."""

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
    """Adopt a detached parent's exact projection or create a foreground one."""

    path = Path(pid_path)
    expected = f"{os.getpid()}\n".encode("ascii")
    with serialized_lock_update(path):
        try:
            payload, evidence = _read_stable_regular_bytes(path, max_bytes=32)
        except _StableArtifactReadError as exc:
            raise ValueError(f"unsafe master PID projection: {exc}") from exc
        if payload is not None:
            if (
                payload != expected
                or int(evidence.get("uid", -1)) != os.geteuid()
                or int(evidence.get("link_count", -1)) != 1
                or not stat.S_ISREG(int(evidence.get("mode", 0)))
                or stat.S_IMODE(int(evidence.get("mode", 0))) != 0o600
            ):
                raise ValueError("master PID projection is not owned by this runner")
            return
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


@dataclass(frozen=True)
class _LgcvfLiveDaemonTerminationAuthority:
    """Parent-held authority for fencing one sealed live daemon generation."""

    profile_id: str
    state_dir: Path
    state_prefix: str
    todo_path: Path
    pid_path: Path
    identity_path: Path
    owner_scope: tuple[tuple[str, str], ...]
    sealed_command_prefix: tuple[str, ...]


def _lgcvf_live_daemon_termination_authority(
    *,
    profile: LifecycleProfile,
    context: LgcvfConfiguredBoardLiveContext,
    supervisor_argv: Sequence[str],
) -> _LgcvfLiveDaemonTerminationAuthority:
    """Bind daemon shutdown authority to the immutable parent launch profile."""

    context = verify_lgcvf_configured_board_live_context(
        capsule_pin_json=context.capsule_pin_json,
        capsule_descriptor=context.capsule_descriptor,
        admission_json=context.admission_json,
        native_launch_json=context.native_launch_json,
        native_descriptor=context.native_descriptor,
    )
    argv = tuple(str(item) for item in supervisor_argv)
    state_dirs = _profile_option_values(argv, "--state-dir")
    state_prefixes = _profile_option_values(argv, "--state-prefix")
    todo_paths = _profile_option_values(argv, "--todo-path")
    shard_counts = _profile_option_values(argv, "--task-shard-count")
    shard_indices = _profile_option_values(argv, "--task-shard-index")
    namespaces = _profile_option_values(argv, "--board-namespace")
    if (
        len(state_dirs) != 1
        or len(state_prefixes) != 1
        or len(todo_paths) != 1
        or shard_counts != ("4",)
        or len(shard_indices) != 1
        or shard_indices[0] not in {"0", "1", "2", "3"}
        or namespaces != (LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,)
        or state_prefixes[0] != f"lgcvf_lane_{shard_indices[0]}"
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", state_prefixes[0])
        is None
    ):
        raise ValueError("LGCVF live daemon termination scope is not exact")
    repository_root = Path(profile.repository_root)
    state_dir = Path(state_dirs[0])
    todo_path = Path(todo_paths[0])
    if (
        not repository_root.is_absolute()
        or repository_root.resolve(strict=False) != repository_root
        or Path(profile.cwd) != repository_root
        or not state_dir.is_absolute()
        or state_dir.resolve(strict=False) != state_dir
        or state_dir != Path(profile.state_root)
        or not _path_within(state_dir, repository_root)
        or not todo_path.is_absolute()
        or todo_path.resolve(strict=False) != todo_path
        or not _path_within(todo_path, repository_root)
    ):
        raise ValueError("LGCVF live daemon termination paths are not exact")
    supervisor_command = tuple(
        build_lgcvf_configured_board_live_module_command(
            python_executable=profile.argv[0],
            capsule_pin_json=context.capsule_pin_json,
            capsule_descriptor=context.capsule_descriptor,
            admission_json=context.admission_json,
            native_launch_json=context.native_launch_json,
            native_descriptor=context.native_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=argv,
        )
    )
    if (
        len(profile.argv) < len(supervisor_command)
        or tuple(profile.argv[-len(supervisor_command) :]) != supervisor_command
    ):
        raise ValueError("LGCVF live supervisor command is outside its profile")
    sealed_command_prefix = tuple(
        build_lgcvf_configured_board_live_module_command(
            python_executable=profile.argv[0],
            capsule_pin_json=context.capsule_pin_json,
            capsule_descriptor=context.capsule_descriptor,
            admission_json=context.admission_json,
            native_launch_json=context.native_launch_json,
            native_descriptor=context.native_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_daemon"
            ),
            argv=(),
        )
    )
    from ..todo_daemon.supervisor_runtime import supervised_child_identity_path

    state_prefix = state_prefixes[0]
    pid_path = state_dir / f"{state_prefix}_managed_daemon.pid"
    identity_path = supervised_child_identity_path(pid_path)
    owner_scope = tuple(
        sorted(
            {
                "repo_root": str(repository_root),
                "state_dir": str(state_dir),
                "state_prefix": state_prefix,
                "todo_path": str(todo_path),
                "daemon_entrypoint": (
                    "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                    "implementation_daemon"
                ),
            }.items()
        )
    )
    return _LgcvfLiveDaemonTerminationAuthority(
        profile_id=profile.profile_id,
        state_dir=state_dir,
        state_prefix=state_prefix,
        todo_path=todo_path,
        pid_path=pid_path,
        identity_path=identity_path,
        owner_scope=owner_scope,
        sealed_command_prefix=sealed_command_prefix,
    )


def _lgcvf_live_boot_id() -> str:
    """Read the Linux boot identity required by a live parent attestation."""

    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as exc:
        raise ProcessIdentityMismatch("Linux boot identity is unavailable") from exc
    if re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        boot_id,
    ) is None:
        raise ProcessIdentityMismatch("Linux boot identity is invalid")
    return boot_id.lower()


def _capture_lgcvf_live_gated_process_identity(
    process: subprocess.Popen[bytes],
    profile: LifecycleProfile,
) -> ProcessIdentity:
    """Attest an unreleased owned gate without ptrace-gated procfs fields.

    The live bootstrap becomes non-dumpable before it waits for the parent's
    release byte.  ``/proc/<pid>/environ``, ``cwd``, and ``exe`` are therefore
    intentionally unavailable.  The parent nevertheless owns an exact Popen
    child which cannot run the requested supervisor yet, and supplied its
    immutable command, cwd, environment, and new-session request itself.  Bind
    that parent-held profile to two stable ``stat`` observations instead of
    weakening the generic marker-observing lifecycle adapter.
    """

    pid = int(process.pid)
    if pid <= 1 or process.poll() is not None:
        raise ProcessIdentityMismatch("LGCVF live gate exited before birth capture")
    popen_args = process.args
    if (
        isinstance(popen_args, (str, bytes))
        or tuple(popen_args) != tuple(profile.argv)
    ):
        raise ProcessIdentityMismatch(
            "LGCVF live gate Popen command differs from its parent-held profile"
        )
    adapter = LinuxProcessAdapter()
    try:
        first = adapter._stat(pid)  # noqa: SLF001 - exact Linux birth fields
        boot_id = _lgcvf_live_boot_id()
        second = adapter._stat(pid)  # noqa: SLF001 - close observation race
    except (OSError, UnicodeError, ValueError, ProcessLookupError) as exc:
        raise ProcessIdentityMismatch(
            "LGCVF live gate process birth is unavailable"
        ) from exc
    if first != second or process.poll() is not None:
        raise ProcessIdentityMismatch(
            "LGCVF live gate process birth changed during capture"
        )
    parent, process_group, session, started = second
    if (
        parent != os.getpid()
        or process_group != pid
        or session != pid
        or started <= 0
    ):
        raise ProcessIdentityMismatch(
            "LGCVF live gate is not the parent's dedicated-session child"
        )
    if not profile.argv or not Path(profile.argv[0]).is_absolute():
        raise ProcessIdentityMismatch("LGCVF live profile executable is not exact")
    executable = Path(profile.argv[0]).resolve(strict=True)
    identity = ProcessIdentity(
        pid=pid,
        start_time_ticks=started,
        parent_pid=parent,
        process_group_id=process_group,
        session_id=session,
        boot_id=boot_id,
        argv=tuple(profile.argv),
        cwd=profile.cwd,
        executable=str(executable),
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=0,
        configuration_root=profile.configuration_root,
    )
    try:
        _parent, _group, _session, final_started = adapter._stat(  # noqa: SLF001
            pid
        )
    except (OSError, UnicodeError, ValueError, ProcessLookupError) as exc:
        raise ProcessIdentityMismatch(
            "LGCVF live gate exited after birth capture"
        ) from exc
    if final_started != identity.start_time_ticks or process.poll() is not None:
        raise ProcessIdentityMismatch(
            "LGCVF live gate process birth changed after capture"
        )
    return identity


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


def start_track(
    track: SupervisorTrack,
    *,
    repo_root: Path,
    common_args: Sequence[str],
    python_executable: str = "python3",
    accepted_control_plane_pin: AgentImplementationControlPlanePin | None = None,
    accepted_control_plane_descriptor: int = -1,
    configured_board_live_context: (
        LgcvfConfiguredBoardLiveContext | None
    ) = None,
    output: OutputFn = _default_output,
) -> subprocess.Popen[bytes]:
    """Start one marker-bound supervisor tree and write its PID projection.

    The PID file remains for legacy observability.  Stop/restart decisions use
    the inherited lifecycle markers and exact OS identities attached to the
    returned process, never the PID projection.
    """

    live_profile_required = _configured_board_live_seal_required(
        common_args,
        (track,),
    )
    if live_profile_required and configured_board_live_context is None:
        raise ValueError(CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_NO_GO)
    if configured_board_live_context is not None:
        if not live_profile_required:
            raise ValueError(
                "LGCVF configured-board live context requires its exact profile"
            )
        configured_board_live_context = (
            verify_lgcvf_configured_board_live_context(
                capsule_pin_json=(
                    configured_board_live_context.capsule_pin_json
                ),
                capsule_descriptor=(
                    configured_board_live_context.capsule_descriptor
                ),
                admission_json=configured_board_live_context.admission_json,
                native_launch_json=(
                    configured_board_live_context.native_launch_json
                ),
                native_descriptor=(
                    configured_board_live_context.native_descriptor
                ),
            )
        )
        _verify_lgcvf_configured_board_live_profile(
            tracks=(track,),
            repo_root=repo_root,
            common_args=common_args,
            context=configured_board_live_context,
            require_complete_lane_set=False,
        )

    resolved = track.resolve(repo_root)
    child_command = (
        [python_executable, "-m", resolved.module_name, *resolved.extra_args]
        if resolved.module_name
        else [python_executable, str(resolved.script_path), *common_args, *resolved.extra_args]
    )
    plan_bound_dispatch = "--plan-bound-dispatch" in resolved.extra_args
    lgcvf_live_dispatch = configured_board_live_context is not None
    if plan_bound_dispatch and lgcvf_live_dispatch:
        raise ValueError("plan-bound and LGCVF live dispatch cannot be combined")
    gate_read_fd: int | None = None
    gate_write_fd: int | None = None
    recovery_authorization_cid = ""
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
            python_executable=python_executable,
            pin=accepted_control_plane_pin,
            descriptor=accepted_control_plane_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=supervisor_argv,
        )
        gate_argv = [
            PLAN_BOUND_LAUNCH_GATE_MARKER,
            str(gate_read_fd),
            str(accepted_tree_root),
            accepted_control_plane_pin_json(accepted_control_plane_pin),
            str(accepted_control_plane_descriptor),
            recovery_authorization_cid or "-",
            "--",
            *child_command,
        ]
        command = build_sealed_control_plane_module_command(
            python_executable=python_executable,
            pin=accepted_control_plane_pin,
            descriptor=accepted_control_plane_descriptor,
            module_name=PLAN_BOUND_LAUNCH_GATE_MODULE,
            argv=gate_argv,
        )
    elif lgcvf_live_dispatch:
        assert configured_board_live_context is not None
        gate_read_fd, gate_write_fd = os.pipe()
        supervisor_argv = [
            *common_args,
            *resolved.extra_args,
            _LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_FLAG,
            LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH,
            "--configured-board-live-capsule-pin-json",
            configured_board_live_context.capsule_pin_json,
            "--configured-board-live-capsule-fd",
            str(configured_board_live_context.capsule_descriptor),
            "--configured-board-live-admission-json",
            configured_board_live_context.admission_json,
            "--configured-board-live-native-launch-json",
            configured_board_live_context.native_launch_json,
            "--configured-board-live-native-fd",
            str(configured_board_live_context.native_descriptor),
        ]
        child_command = build_lgcvf_configured_board_live_module_command(
            python_executable=python_executable,
            capsule_pin_json=(
                configured_board_live_context.capsule_pin_json
            ),
            capsule_descriptor=(
                configured_board_live_context.capsule_descriptor
            ),
            admission_json=configured_board_live_context.admission_json,
            native_launch_json=(
                configured_board_live_context.native_launch_json
            ),
            native_descriptor=(
                configured_board_live_context.native_descriptor
            ),
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=supervisor_argv,
        )
        gate_argv = [
            LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_MARKER,
            str(gate_read_fd),
            str(accepted_tree_root),
            configured_board_live_context.capsule_pin_json,
            str(configured_board_live_context.capsule_descriptor),
            configured_board_live_context.admission_json,
            configured_board_live_context.native_launch_json,
            str(configured_board_live_context.native_descriptor),
            "--",
            *child_command,
        ]
        command = build_lgcvf_configured_board_live_module_command(
            python_executable=python_executable,
            capsule_pin_json=(
                configured_board_live_context.capsule_pin_json
            ),
            capsule_descriptor=(
                configured_board_live_context.capsule_descriptor
            ),
            admission_json=configured_board_live_context.admission_json,
            native_launch_json=(
                configured_board_live_context.native_launch_json
            ),
            native_descriptor=(
                configured_board_live_context.native_descriptor
            ),
            module_name=LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_MODULE,
            argv=gate_argv,
        )
    pid_reservation_fd: int | None = None
    pid_reservation_identity: tuple[int, int] | None = None
    live_daemon_termination_authority: (
        _LgcvfLiveDaemonTerminationAuthority | None
    ) = None
    configuration_root = "sha256:" + hashlib.sha256(
        json.dumps(
            command, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    state_root = resolved.supervisor_pid_path.parent.resolve(strict=False)
    run_root = state_root / "lifecycle-runs" / resolved.name
    status_path = _inferred_supervisor_status_path(resolved)
    profile_environment_values = dict(
        _lgcvf_configured_board_live_profile_environment(os.environ)
        if lgcvf_live_dispatch
        else (
            _plan_bound_profile_environment(os.environ)
            if plan_bound_dispatch
            else ()
        )
    )
    if not lgcvf_live_dispatch:
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
    launch_environment = profile.launch_environment(0)
    if plan_bound_dispatch or lgcvf_live_dispatch:
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
            QUACK_TOKEN_FILE_ENV,
            BOARD_EXTENSION_INSTALL_POLICY_ENV,
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
        launch_environment = (
            _lgcvf_configured_board_live_positive_child_environment(
                launch_environment,
                common_args=common_args,
            )
            if lgcvf_live_dispatch
            else _plan_bound_positive_child_environment(launch_environment)
        )
    if lgcvf_live_dispatch:
        assert configured_board_live_context is not None
        configured_board_live_context = verify_lgcvf_configured_board_live_context(
            capsule_pin_json=configured_board_live_context.capsule_pin_json,
            capsule_descriptor=configured_board_live_context.capsule_descriptor,
            admission_json=configured_board_live_context.admission_json,
            native_launch_json=configured_board_live_context.native_launch_json,
            native_descriptor=configured_board_live_context.native_descriptor,
        )
        live_daemon_termination_authority = (
            _lgcvf_live_daemon_termination_authority(
                profile=profile,
                context=configured_board_live_context,
                supervisor_argv=supervisor_argv,
            )
        )
    resolved.log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.supervisor_pid_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_bound_dispatch or lgcvf_live_dispatch:
        (
            pid_reservation_fd,
            pid_reservation_identity,
        ) = _reserve_owned_pid_projection(resolved.supervisor_pid_path)
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
    try:
        try:
            if lgcvf_live_dispatch:
                assert configured_board_live_context is not None
                configured_board_live_context = (
                    verify_lgcvf_configured_board_live_context(
                        capsule_pin_json=(
                            configured_board_live_context.capsule_pin_json
                        ),
                        capsule_descriptor=(
                            configured_board_live_context.capsule_descriptor
                        ),
                        admission_json=(
                            configured_board_live_context.admission_json
                        ),
                        native_launch_json=(
                            configured_board_live_context.native_launch_json
                        ),
                        native_descriptor=(
                            configured_board_live_context.native_descriptor
                        ),
                    )
                )
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=launch_environment,
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(
                    tuple(
                        sorted(
                            {
                                gate_read_fd,
                                *(
                                    configured_board_live_context.pass_fds
                                    if lgcvf_live_dispatch
                                    and configured_board_live_context is not None
                                    else (accepted_control_plane_descriptor,)
                                ),
                            }
                        )
                    )
                    if (
                        (plan_bound_dispatch or lgcvf_live_dispatch)
                        and gate_read_fd is not None
                    )
                    else ()
                ),
            )
        except BaseException:
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
    if gate_read_fd is not None:
        os.close(gate_read_fd)
    # Popen is only an observation handle.  The immutable profile is what lets
    # stop/restart rediscover children that have detached or been reparented.
    process._agent_supervisor_lifecycle_profile = profile
    if lgcvf_live_dispatch:
        if live_daemon_termination_authority is None:
            raise AssertionError("LGCVF live daemon termination authority is absent")
        process._agent_supervisor_live_daemon_termination_authority = (
            live_daemon_termination_authority
        )
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
            raise PlanBoundProcessBirthError(
                "plan-bound process birth capture failed; launch remained gated",
                pid=int(process.pid),
                profile=profile,
                all_trees_fenced=all_trees_fenced,
            ) from exc
        finally:
            if gate_write_fd is not None:
                os.close(gate_write_fd)
    elif lgcvf_live_dispatch:
        if gate_write_fd is None or configured_board_live_context is None:
            raise AssertionError("LGCVF live launch gate was not created")
        try:
            configured_board_live_context = (
                verify_lgcvf_configured_board_live_context(
                    capsule_pin_json=(
                        configured_board_live_context.capsule_pin_json
                    ),
                    capsule_descriptor=(
                        configured_board_live_context.capsule_descriptor
                    ),
                    admission_json=(
                        configured_board_live_context.admission_json
                    ),
                    native_launch_json=(
                        configured_board_live_context.native_launch_json
                    ),
                    native_descriptor=(
                        configured_board_live_context.native_descriptor
                    ),
                )
            )
            process_identity = _capture_lgcvf_live_gated_process_identity(
                process,
                profile,
            )
            if not isinstance(process_identity, ProcessIdentity):
                raise ProcessIdentityMismatch(
                    "LGCVF live launch returned no typed process identity"
                )
            process._agent_supervisor_process_identity = process_identity
            process._agent_supervisor_live_admission_id = (
                configured_board_live_context.admission.admission_id
            )
            process._agent_supervisor_live_capsule_id = getattr(
                configured_board_live_context.capsule_pin,
                "capsule_id",
            )
            if (
                pid_reservation_fd is None
                or pid_reservation_identity is None
            ):
                raise AssertionError("LGCVF live PID projection was not reserved")
            _publish_reserved_pid_projection(
                resolved.supervisor_pid_path,
                pid_reservation_fd,
                pid_reservation_identity,
                int(process.pid),
            )
            os.close(pid_reservation_fd)
            pid_reservation_fd = None
            if os.write(
                gate_write_fd,
                LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_SUCCESS,
            ) != len(LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_SUCCESS):
                raise OSError("LGCVF live launch gate release was incomplete")
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
            raise LgcvfConfiguredBoardLiveProcessBirthError(
                "LGCVF live process birth capture failed; launch remained gated",
                pid=int(process.pid),
                profile=profile,
                all_trees_fenced=all_trees_fenced,
            ) from exc
        finally:
            if gate_write_fd is not None:
                os.close(gate_write_fd)
    else:
        try:
            process_identity = LinuxProcessAdapter()._identity(  # noqa: SLF001
                int(process.pid), profile
            )
        except (
            OSError,
            UnicodeError,
            ValueError,
            ProcessLookupError,
            ProcessIdentityMismatch,
        ):
            # Legacy tracks retain their previous best-effort observability.
            process_identity = None
        process._agent_supervisor_process_identity = process_identity
        resolved.supervisor_pid_path.write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
    _emit(
        output,
        f"started {resolved.name} supervisor pid={process.pid} script={resolved.script_path} log={resolved.log_path}",
    )
    return process


def _fence_unreleased_plan_bound_process(
    process: subprocess.Popen[bytes],
) -> bool:
    """Reap a still-gated root that was never allowed to exec its child.

    Before the gate byte, the accepted-tree helper performs no fork or exec.
    Closing its only authorization writer therefore makes the exact
    marker-bound tree contain only this owned Popen root, which is then reaped
    synchronously (and terminated/killed if it does not consume EOF).
    """

    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                return False
    return process.poll() is not None


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


def _lgcvf_live_exact_root_state(identity: ProcessIdentity) -> str:
    """Return alive/dead/unknown for one exact non-dumpable live root."""

    try:
        current_boot_id = _lgcvf_live_boot_id()
    except ProcessIdentityMismatch:
        return "unknown"
    if identity.boot_id != current_boot_id:
        return "dead"
    try:
        _parent, group, session, started = LinuxProcessAdapter._stat(  # noqa: SLF001
            identity.pid
        )
    except (FileNotFoundError, ProcessLookupError):
        return "dead"
    except (OSError, UnicodeError, ValueError):
        return "unknown"
    if started != identity.start_time_ticks:
        return "dead"
    if (
        group != identity.process_group_id
        or session != identity.session_id
    ):
        return "unknown"
    return "alive"


def _lgcvf_live_daemon_command_matches(
    command: Sequence[str],
    authority: _LgcvfLiveDaemonTerminationAuthority,
) -> bool:
    """Verify a sidecar command against the exact sealed daemon and lane scope."""

    tokens = tuple(str(item) for item in command)
    prefix = authority.sealed_command_prefix
    expected_shard_index = authority.state_prefix.removeprefix("lgcvf_lane_")
    try:
        return bool(
            prefix
            and expected_shard_index in {"0", "1", "2", "3"}
            and len(tokens) > len(prefix)
            and tokens[: len(prefix)] == prefix
            and _profile_option_values(tokens, "--state-dir")
            == (str(authority.state_dir),)
            and _profile_option_values(tokens, "--state-prefix")
            == (authority.state_prefix,)
            and _profile_option_values(tokens, "--todo-path")
            == (str(authority.todo_path),)
            and _profile_option_values(tokens, "--board-namespace")
            == (LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,)
            and _profile_option_values(tokens, "--task-shard-count") == ("4",)
            and _profile_option_values(tokens, "--task-shard-index")
            == (expected_shard_index,)
            and tokens.count("--strict-task-sharding") == 1
        )
    except ValueError:
        return False


def _fence_lgcvf_live_daemon_sidecar(
    authority: _LgcvfLiveDaemonTerminationAuthority,
    *,
    grace_seconds: float,
) -> tuple[bool, tuple[int, ...]]:
    """Fence the exact daemon group named by its content-addressed sidecar."""

    from ..merge.worktree_lifecycle import OwnerLiveness, read_process_birth
    from ..todo_daemon.core import terminate_pid_tree
    from ..todo_daemon.supervisor_runtime import (
        SupervisedChildIdentity,
        read_process_command_argv,
        supervised_child_identity_liveness,
    )

    try:
        payload, evidence = _read_stable_regular_json(
            authority.identity_path,
            max_bytes=262_144,
        )
    except _StableArtifactReadError:
        return False, ()
    if payload is None:
        # The content-addressed identity is the signal authority.  A raw PID
        # without it is deliberately not enough to claim that no daemon exists.
        try:
            raw_pid, _pid_evidence = _read_stable_regular_bytes(
                authority.pid_path,
                max_bytes=64,
            )
        except _StableArtifactReadError:
            return False, ()
        return (raw_pid is None), ()
    if evidence.get("uid") != os.geteuid():
        return False, ()
    identity = SupervisedChildIdentity.from_dict(payload)
    if (
        identity is None
        or dict(identity.owner_scope) != dict(authority.owner_scope)
        or not _lgcvf_live_daemon_command_matches(
            identity.command,
            authority,
        )
    ):
        return False, ()
    daemon_pid = int(identity.process_birth.pid)
    try:
        raw_pid, pid_evidence = _read_stable_regular_bytes(
            authority.pid_path,
            max_bytes=64,
        )
    except _StableArtifactReadError:
        return False, ()
    if raw_pid is not None:
        try:
            projected_pid = int(raw_pid.decode("ascii").strip())
        except (UnicodeError, ValueError):
            return False, ()
        if (
            projected_pid != daemon_pid
            or pid_evidence.get("uid") != os.geteuid()
        ):
            return False, ()
    liveness = supervised_child_identity_liveness(identity)
    if liveness is OwnerLiveness.UNKNOWN:
        return False, ()
    if liveness is OwnerLiveness.ALIVE:
        observed_argv = read_process_command_argv(daemon_pid)
        if observed_argv != identity.command:
            return False, ()
        # Close the command-read race with the same durable birth check used by
        # the supervisor runtime immediately before entering the strict fence.
        if supervised_child_identity_liveness(identity) is not OwnerLiveness.ALIVE:
            return False, ()
    else:
        # DEAD has two materially different meanings.  If this numeric PID now
        # has another birth (or the machine rebooted), the durable owner is
        # conclusively gone and its old process group cannot still exist; do
        # not present the reused PID to a signal path.  If the leader is simply
        # absent, retain the captured PGID authority so strict fencing can find
        # and kill surviving members of the daemon's dedicated group.
        try:
            current_boot_id = _lgcvf_live_boot_id()
            current_birth = read_process_birth(daemon_pid)
        except (OSError, ProcessIdentityMismatch):
            return False, ()
        if (
            identity.process_birth.boot_id
            and identity.process_birth.boot_id != current_boot_id
        ):
            return True, (daemon_pid,)
        if current_birth is not None:
            if (
                current_birth.start_time_ticks
                != identity.process_birth.start_time_ticks
                or (
                    identity.process_birth.boot_id
                    and current_birth.boot_id
                    and current_birth.boot_id
                    != identity.process_birth.boot_id
                )
            ):
                return True, (daemon_pid,)
            # A supposedly DEAD identity that still has the exact live birth is
            # contradictory inspection evidence, so fail closed.
            return False, ()
    fenced = terminate_pid_tree(
        daemon_pid,
        grace_seconds=max(0.0, float(grace_seconds)),
        freeze_first=True,
        require_gone=True,
        owned_process_group_id=daemon_pid,
        expected_root_start_time_ticks=(
            identity.process_birth.start_time_ticks
        ),
    )
    gone = supervised_child_identity_liveness(identity)
    return bool(fenced and gone is OwnerLiveness.DEAD), (daemon_pid,)


def _terminate_lgcvf_live_managed_process(
    process: subprocess.Popen[bytes],
    *,
    profile: LifecycleProfile,
    process_identity: ProcessIdentity,
    authority: _LgcvfLiveDaemonTerminationAuthority,
    grace_seconds: float,
) -> tuple[bool, tuple[int, ...]]:
    """Strictly fence one non-dumpable live root and its sealed daemon group."""

    from ..todo_daemon.core import terminate_pid_tree

    if (
        authority.profile_id != profile.profile_id
        or process_identity.pid != int(process.pid)
        or process_identity.profile_id != profile.profile_id
        or process_identity.run_id != profile.run_id
        or process_identity.target_id != profile.target_id
        or process_identity.state_root != profile.state_root
        or process_identity.configuration_root != profile.configuration_root
    ):
        return False, ()
    root_state = _lgcvf_live_exact_root_state(process_identity)
    if root_state == "unknown":
        # Inaccessible expected live state is never collapsed into absence.
        return False, ()
    stopped: set[int] = set()
    if root_state == "alive":
        if not terminate_pid_tree(
            process_identity.pid,
            grace_seconds=max(0.0, float(grace_seconds)),
            freeze_first=True,
            require_gone=True,
            owned_process_group_id=process_identity.process_group_id,
            expected_root_start_time_ticks=process_identity.start_time_ticks,
        ):
            return False, ()
        stopped.add(process_identity.pid)
        if _lgcvf_live_exact_root_state(process_identity) != "dead":
            return False, tuple(sorted(stopped))
    daemon_fenced, daemon_pids = _fence_lgcvf_live_daemon_sidecar(
        authority,
        grace_seconds=grace_seconds,
    )
    stopped.update(daemon_pids)
    return daemon_fenced, tuple(sorted(stopped))


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
    live_admission_id = getattr(
        process,
        "_agent_supervisor_live_admission_id",
        "",
    )
    live_capsule_id = getattr(
        process,
        "_agent_supervisor_live_capsule_id",
        "",
    )
    live_authority = getattr(
        process,
        "_agent_supervisor_live_daemon_termination_authority",
        None,
    )
    if live_admission_id or live_capsule_id or live_authority is not None:
        process_identity = getattr(
            process,
            "_agent_supervisor_process_identity",
            None,
        )
        if (
            not isinstance(live_admission_id, str)
            or not live_admission_id
            or not isinstance(live_capsule_id, str)
            or not live_capsule_id
            or not isinstance(process_identity, ProcessIdentity)
            or not isinstance(
                live_authority,
                _LgcvfLiveDaemonTerminationAuthority,
            )
        ):
            return False, ()
        return _terminate_lgcvf_live_managed_process(
            process,
            profile=profile,
            process_identity=process_identity,
            authority=live_authority,
            grace_seconds=grace_seconds,
        )
    adapter = LinuxProcessAdapter()
    tree = adapter.snapshot(profile)
    if not tree.members:
        return True, ()
    root_ids = {item.pid for item in tree.roots}
    process_member = next(
        (item for item in tree.members if item.pid == process.pid), None
    )
    if process_member is not None and process.pid not in root_ids:
        raise ProcessIdentityMismatch(
            "managed Popen does not identify the marker-bound tree root"
        )
    member_pids = tuple(item.pid for item in tree.members)
    adapter.terminate(
        tree,
        grace_seconds=grace_seconds,
        deadline_ms=max(1, int(max(0.0, grace_seconds) * 1000) + 1_000),
    )
    deadline = time.monotonic() + max(0.1, grace_seconds) + 1.0
    while time.monotonic() < deadline:
        if not any(adapter.identity_alive(item) for item in tree.members):
            if not adapter.snapshot(profile).members:
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
    """Stop exact marker-bound wrapper trees and verify no descendants remain."""

    stopped: list[int] = []
    removed_runtime_markers: list[str] = []
    all_fenced = True
    _emit(output, "stopping supervisor wrapper and managed daemons")
    for track in tracks:
        process = processes.get(track.name)
        fenced, member_pids = _terminate_managed_process(
            process,
            grace_seconds=grace_seconds,
        )
        if fenced:
            stopped.extend(member_pids)
        elif process is not None:
            all_fenced = False
            _emit(
                output,
                f"could not verify complete shutdown for {track.name} pid={process.pid}",
            )
        if process is not None:
            try:
                process.wait(timeout=max(0.1, grace_seconds))
            except subprocess.TimeoutExpired:
                pass
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
    require_lgcvf_configured_board_live_seal: str = "",
    configured_board_live_capsule_pin_json: str = "",
    configured_board_live_capsule_fd: int = -1,
    configured_board_live_admission_json: str = "",
    configured_board_live_native_launch_json: str = "",
    configured_board_live_native_fd: int = -1,
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
    lgcvf_live_config = str(
        require_lgcvf_configured_board_live_seal or ""
    )
    lgcvf_profile_requested = _profile_option_values(
        tuple(str(item) for item in common_args),
        "--board-namespace",
    ) == (LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,)
    live_binding_presence = (
        bool(configured_board_live_capsule_pin_json),
        configured_board_live_capsule_fd >= 3,
        bool(configured_board_live_admission_json),
        bool(configured_board_live_native_launch_json),
        configured_board_live_native_fd >= 3,
    )
    has_complete_live_binding = all(live_binding_presence)
    if any(live_binding_presence) and not has_complete_live_binding:
        raise ValueError("LGCVF configured-board live binding is incomplete")
    if lgcvf_profile_requested != bool(lgcvf_live_config):
        raise ValueError(
            "LGCVF configured-board profile and live-seal flag are bidirectional"
        )
    if bool(lgcvf_live_config) != has_complete_live_binding:
        raise ValueError(
            "LGCVF configured-board live flag and descriptors are bidirectional"
        )
    configured_board_live_context: (
        LgcvfConfiguredBoardLiveContext | None
    ) = None
    if lgcvf_live_config:
        relative = _configured_board_gate_relative_path(
            lgcvf_live_config,
            field="LGCVF configured-board live-seal config",
        )
        if relative != LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH:
            raise ValueError(
                "LGCVF configured-board live seal requires its candidate config"
            )
        if not live_profile_required:
            raise ValueError(
                "LGCVF configured-board live seal requires datasets authority"
            )
        if plan_bound_children or accepted_control_plane_pin is not None:
            raise ValueError(
                "LGCVF configured-board live seal cannot reuse plan-bound authority"
            )
        configured_board_live_context = (
            verify_lgcvf_configured_board_live_context(
                capsule_pin_json=configured_board_live_capsule_pin_json,
                capsule_descriptor=configured_board_live_capsule_fd,
                admission_json=configured_board_live_admission_json,
                native_launch_json=configured_board_live_native_launch_json,
                native_descriptor=configured_board_live_native_fd,
            )
        )
        _verify_lgcvf_configured_board_live_profile(
            tracks=managed_tracks,
            repo_root=repo_root,
            common_args=common_args,
            context=configured_board_live_context,
        )
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
        if plan_bound_children or configured_board_live_context is not None:
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
            configured_board_live_context=configured_board_live_context,
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
    except (
        PlanBoundProcessBirthError,
        LgcvfConfiguredBoardLiveProcessBirthError,
    ) as exc:
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
        "configured_board_live_admission_id": (
            configured_board_live_context.admission.admission_id
            if configured_board_live_context is not None
            else ""
        ),
        "configured_board_live_capsule_id": (
            str(
                getattr(
                    configured_board_live_context.capsule_pin,
                    "capsule_id",
                    "",
                )
            )
            if configured_board_live_context is not None
            else ""
        ),
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
    parser.add_argument(
        "--require-lgcvf-configured-board-live-seal",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-capsule-pin-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-capsule-fd",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-admission-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-native-launch-json",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--configured-board-live-native-fd",
        type=int,
        default=-1,
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
    if getattr(args, "require_lgcvf_configured_board_live_seal", ""):
        raise ValueError(
            "LGCVF configured-board live runner must retain its sealed parent FDs"
        )

    master_log, master_pid = _master_paths(args)
    master_log.parent.mkdir(parents=True, exist_ok=True)
    master_pid.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
        *_without_detach(argv),
    ]
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
                    stdin=subprocess.DEVNULL,
                    stdout=out_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                out_handle.close()
            _publish_reserved_pid_projection(
                master_pid,
                descriptor,
                reservation_identity,
                int(process.pid),
            )
        except BaseException:
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

    # Wrapper controls may describe the same child policy already present in
    # the explicit common profile.  Suppress only that derived copy; retain
    # the explicit argv unchanged so ambiguous raw duplicates remain visible
    # to fail-closed profile validation.
    explicit_common_args = [str(item) for item in args.common_arg]
    explicit_strict_task_sharding = (
        "--strict-task-sharding" in explicit_common_args
    )
    explicit_idle_lane_work_stealing = (
        "--idle-lane-work-stealing" in explicit_common_args
    )
    strict_task_sharding = bool(
        getattr(
            args,
            "implementation_supervisor_strict_task_sharding",
            False,
        )
    )
    idle_lane_work_stealing = str(
        getattr(
            args,
            "implementation_supervisor_idle_lane_work_stealing",
            "",
        )
        or ""
    )
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
                strict_task_sharding=(
                    strict_task_sharding
                    and not explicit_strict_task_sharding
                ),
                idle_lane_work_stealing=(
                    ""
                    if explicit_idle_lane_work_stealing
                    else idle_lane_work_stealing
                ),
            )
        )
    if (
        strict_task_sharding
        and "--strict-task-sharding" not in common_args
        and not explicit_strict_task_sharding
    ):
        common_args.append("--strict-task-sharding")
    if (
        idle_lane_work_stealing
        and "--idle-lane-work-stealing" not in common_args
        and not explicit_idle_lane_work_stealing
    ):
        common_args.extend(
            ["--idle-lane-work-stealing", idle_lane_work_stealing]
        )
    common_args.extend(explicit_common_args)
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


def _run_lgcvf_configured_board_live_launch_gate(
    argv: Sequence[str],
) -> int:
    """Release one capsule/native-bound supervisor after parent birth capture."""

    tokens = tuple(str(item) for item in argv)
    if len(tokens) < 10 or tokens[7] != "--":
        return 78
    try:
        gate_fd = int(tokens[0])
        capsule_fd = int(tokens[3])
        native_fd = int(tokens[6])
        context = verify_lgcvf_configured_board_live_context(
            capsule_pin_json=tokens[2],
            capsule_descriptor=capsule_fd,
            admission_json=tokens[4],
            native_launch_json=tokens[5],
            native_descriptor=native_fd,
        )
        accepted_tree_root = _canonical_accepted_tree_root(Path(tokens[1]))
    except (OSError, ValueError):
        return 78
    child_command = list(tokens[8:])
    try:
        expected_prefix = build_lgcvf_configured_board_live_module_command(
            python_executable=child_command[0],
            capsule_pin_json=context.capsule_pin_json,
            capsule_descriptor=context.capsule_descriptor,
            admission_json=context.admission_json,
            native_launch_json=context.native_launch_json,
            native_descriptor=context.native_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=(),
        )
    except (IndexError, OSError, ValueError):
        return 78
    prefix_length = len(expected_prefix)
    if child_command[:prefix_length] != expected_prefix:
        return 78
    child_argv = child_command[prefix_length:]
    try:
        exact_options = {
            _LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_FLAG: (
                LGCVF_CONFIGURED_BOARD_LIVE_CONFIG_PATH
            ),
            "--configured-board-live-capsule-pin-json": (
                context.capsule_pin_json
            ),
            "--configured-board-live-capsule-fd": str(
                context.capsule_descriptor
            ),
            "--configured-board-live-admission-json": context.admission_json,
            "--configured-board-live-native-launch-json": (
                context.native_launch_json
            ),
            "--configured-board-live-native-fd": str(
                context.native_descriptor
            ),
            "--board-namespace": LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,
            "--task-source-kind": TASK_SOURCE_DUCKDB,
            "--authority-mode": AUTHORITY_MODE_QUACK,
            "--state-schema-revision": (
                DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA_REVISION
            ),
            "--idle-lane-work-stealing": "virgin-transfer",
            "--task-shard-count": "4",
        }
        for option, expected in exact_options.items():
            _lgcvf_live_exact_option(child_argv, option, expected)
        shard_indices = _profile_option_values(
            child_argv,
            "--task-shard-index",
        )
        state_prefixes = _profile_option_values(child_argv, "--state-prefix")
        if (
            len(shard_indices) != 1
            or shard_indices[0] not in {"0", "1", "2", "3"}
            or state_prefixes != (f"lgcvf_lane_{shard_indices[0]}",)
            or gate_fd < 3
            or gate_fd in context.pass_fds
        ):
            return 78
    except ValueError:
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
    if authorization != LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_SUCCESS:
        return 78
    try:
        verify_lgcvf_configured_board_live_context(
            capsule_pin_json=context.capsule_pin_json,
            capsule_descriptor=context.capsule_descriptor,
            admission_json=context.admission_json,
            native_launch_json=context.native_launch_json,
            native_descriptor=context.native_descriptor,
        )
        environment = (
            _lgcvf_configured_board_live_positive_child_environment(
                os.environ,
                common_args=child_argv,
            )
        )
        environment[REPOSITORY_ROOT_ENV] = str(accepted_tree_root)
        os.execvpe(child_command[0], child_command, environment)
    except (OSError, ValueError):
        return 78
    return 78


def _run_plan_bound_launch_gate(argv: Sequence[str]) -> int:
    """Release exactly one accepted-tree child after parent birth capture."""

    tokens = tuple(str(item) for item in argv)
    if len(tokens) < 8 or tokens[5] != "--":
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
    child_command = list(tokens[6:])
    try:
        expected_prefix = build_sealed_control_plane_module_command(
            python_executable=child_command[0],
            pin=control_plane_pin,
            descriptor=control_plane_descriptor,
            module_name=(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon."
                "implementation_supervisor"
            ),
            argv=(),
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
        environment = _plan_bound_positive_child_environment(os.environ)
        os.execvpe(child_command[0], child_command, environment)
    except OSError:
        return 78
    return 78


def main(argv: list[str] | None = None) -> int:
    from .process_security import (
        capture_state_authority_credentials,
        harden_state_authority_process,
    )

    harden_state_authority_process()
    capture_state_authority_credentials()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list[:1] == [LGCVF_CONFIGURED_BOARD_LIVE_LAUNCH_GATE_MARKER]:
        return _run_lgcvf_configured_board_live_launch_gate(args_list[1:])
    if args_list[:1] == [CONFIGURED_BOARD_LIVE_SEAL_LAUNCH_GATE_MARKER]:
        return 78
    if args_list[:1] == [PLAN_BOUND_LAUNCH_GATE_MARKER]:
        return _run_plan_bound_launch_gate(args_list[1:])
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
    tracks = tracks_from_parsed_args(args)
    effective_common_args = common_args_from_parsed_args(args)
    live_namespace_requested = _profile_option_values(
        effective_common_args,
        "--board-namespace",
    ) == (LGCVF_CONFIGURED_BOARD_LIVE_NAMESPACE,)
    if live_namespace_requested or args.require_lgcvf_configured_board_live_seal:
        try:
            if not args.require_lgcvf_configured_board_live_seal:
                raise ValueError(
                    "LGCVF configured-board profile requires its live-seal flag"
                )
            context = verify_lgcvf_configured_board_live_context(
                capsule_pin_json=args.configured_board_live_capsule_pin_json,
                capsule_descriptor=args.configured_board_live_capsule_fd,
                admission_json=args.configured_board_live_admission_json,
                native_launch_json=(
                    args.configured_board_live_native_launch_json
                ),
                native_descriptor=args.configured_board_live_native_fd,
            )
            _verify_lgcvf_configured_board_live_profile(
                tracks=tracks,
                repo_root=args.repo_root,
                common_args=effective_common_args,
                context=context,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"LGCVF configured-board live binding is invalid: {exc}")
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
            common_args=effective_common_args,
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
            require_lgcvf_configured_board_live_seal=(
                args.require_lgcvf_configured_board_live_seal
            ),
            configured_board_live_capsule_pin_json=(
                args.configured_board_live_capsule_pin_json
            ),
            configured_board_live_capsule_fd=(
                args.configured_board_live_capsule_fd
            ),
            configured_board_live_admission_json=(
                args.configured_board_live_admission_json
            ),
            configured_board_live_native_launch_json=(
                args.configured_board_live_native_launch_json
            ),
            configured_board_live_native_fd=(
                args.configured_board_live_native_fd
            ),
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
