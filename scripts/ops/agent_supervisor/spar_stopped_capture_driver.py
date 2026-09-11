"""Retain new stopped-state custody across explicit reviewable commands.

This driver has no live-capture reconstruction route and never starts a native
owner. Its JSON outputs are audit records; only its in-memory producer object
can reach the distinct stopped-state installer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

from . import spar_capture_runtime as dependencies
from . import spar_stopped_capture as stopped
from . import spar_stopped_origin as origin
from . import spar_merge_owner as role
from . import spar_retained_capture_driver as workflow

SCHEMA = 'spar/stopped-capture-driver@1'
REQUEST_SCHEMA = 'spar/stopped-capture-driver-request@1'
FIELDS = {'schema','repository_root','config_path','fleet_config','controller_birth','helper_birth',
    'controller_argv','hold_sha256','source_head','source_tree','operation_root',
    'runtime_manifest','runtime_manifest_sha256','runtime_helper_sha256','owner_identity'}


class StoppedCaptureDriver:
    def __init__(self, request):
        role._closed(request, FIELDS)
        stopped.require(request['schema'] == REQUEST_SCHEMA, 'stopped driver request schema differs')
        self.request = json.loads(role._json(request))
        self.runtime = dependencies.admit_runtime(Path(request['runtime_manifest']), request['runtime_manifest_sha256'])
        stopped.require(hashlib.sha256(Path(dependencies.__file__).read_bytes()).hexdigest()
            == request['runtime_helper_sha256'], 'stopped runtime helper differs')
        self.output = Path(request['operation_root']).absolute()
        workflow._owned_directory(self.output, create=True)
        self.sequence = 0
        self.stage = 'inspecting'
        self.session = self.captured = self.prepared = self.installed = None
        self.finish_recorded = False
        self.succession = stopped.KeeperSuccession(
            repository_root=request['repository_root'], config_path=request['config_path'],
            controller_birth=request['controller_birth'], helper_birth=request['helper_birth'],
            controller_argv=request['controller_argv'], hold_sha256=request['hold_sha256'],
            source_head=request['source_head'], source_tree=request['source_tree'],
            journal_root=self.output/'succession', fleet_config=request['fleet_config'], owner_identity=request['owner_identity'],
            task_runtime={'manifest_path':request['runtime_manifest'],
                'manifest_sha256':request['runtime_manifest_sha256'], 'helper_sha256':request['runtime_helper_sha256']})
        self.stage = 'inspected'
        self.record('inspected')

    def result(self, **extra):
        return {'schema':SCHEMA,'stage':self.stage,'request_cid':role._cid(self.request),
            'keeper_stage':self.succession.stage,'callback_settled':False,
            'source_admitted':False,'successor_started':False,'completion_authority':False,**extra}

    def record(self, phase):
        self.sequence += 1
        workflow._exclusive_write(self.output/f'{self.sequence:03d}-{phase}.json',role._json(self.result()))

    def require_runtime(self):
        self.runtime.require_current()
        stopped.require(hashlib.sha256(Path(dependencies.__file__).read_bytes()).hexdigest()
            == self.request['runtime_helper_sha256'], 'stopped runtime helper changed')

    def overlap(self):
        stopped.require(self.stage=='inspected','stopped overlap requires exact inspected workflow')
        self.require_runtime()
        self.stage='overlapping'
        self.succession.overlap()
        self.stage='overlapped';self.record('overlapped')
        return self.result()

    def abort_overlap(self):
        stopped.require(self.stage in ('overlapping','overlapped'),'stopped overlap cannot abort after retirement')
        self.succession.abort_overlap()
        self.stage='inspected';self.record('overlap-aborted')
        return self.result()

    def retire(self):
        stopped.require(self.stage in ('overlapped','retiring'),'stopped retirement requires retained overlap')
        self.require_runtime()
        self.stage='retiring'
        try:
            if self.succession.stage=='overlapped':self.succession.retire()
            else:self.succession.complete_retirement(timeout=1)
        finally:
            self.stage = ('retiring' if self.succession.stage == 'retired'
                          and not self.succession.retirement_recorded else self.succession.stage)
        self.stage='retired';self.record('retired')
        return self.result()

    def fence(self):
        stopped.require(self.stage=='retired','fresh stopped fences require positive workflow exits')
        self.require_runtime()
        self.session=stopped.StoppedCaptureSession(self.succession,fleet_config=self.request['fleet_config'])
        self.stage='fenced';self.record('fenced')
        return self.result()

    def capture(self):
        stopped.require(self.stage=='fenced','stopped capture requires retained fresh fences')
        self.require_runtime()
        # Unique failed copies remain preserved. This session already holds all
        # fences and refuses changed task/queue identities on a later attempt.
        destination=self.output/f'capture-{self.sequence+1:03d}'
        self.record('capture-attempt')
        self.captured=self.session.capture(destination=destination)
        self.stage='captured';self.record('captured')
        return self.result(capture_cid=stopped.evidence_cid(self.captured.receipt))

    def prepare(self):
        stopped.require(self.stage=='captured','stopped prepare requires retained fresh capture')
        self.require_runtime();self.captured.require_current()
        destination=self.output/f'prepared-{self.sequence+1:03d}'
        self.record('prepare-attempt')
        self.prepared=role.prepare_offline_clone(offline_root=self.captured.path,destination=destination,
            manifest=self.captured.receipt['manifest'])
        self.stage='prepared';self.record('prepared')
        return self.result(database_uuid=self.prepared.database_uuid)

    def install(self):
        stopped.require(self.stage=='prepared','stopped install requires retained prepared clone')
        self.require_runtime();self.captured.require_current()
        self.stage='installing';self.record('installing')
        self.installed=origin.install_stopped_queue(self.captured,self.prepared)
        self.stage='installed';self.record('installed')
        return self.result(installed=self.installed)

    def finish(self):
        stopped.require(self.stage in ('installed','finishing','finished'),
                        'stopped finish requires successful installation')
        self.require_runtime()
        if self.stage == 'installed':
            stopped.require(self.session._closed_gate()==self.captured.receipt['closure'],
                            'stopped closure changed after install')
            marker=stopped.native._json(self.session.queue_root/origin.REQUIRED_MARKER)
            stopped.require(marker.get('origin_cid')==self.installed['origin_cid']
                and marker.get('capture_cid')==stopped.evidence_cid(self.captured.receipt),'stopped installed marker differs')
            for entry in self.captured.receipt['manifest']['files']:
                role.copy_entry(self.captured.path,entry,None,digest_only=True)
            self.record('finish-prepared')
            self.stage='finishing'
        if self.stage == 'finishing':
            if not self.session._closed:
                self.session.resources.close();self.session._closed=True
            self.succession.close_after_install()
            self.stage='finished'
        if not self.finish_recorded:
            self.record('finished')
            self.finish_recorded=True
        return self.result()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args()
    raw=stopped.native._bytes(Path(args.request))
    stopped.require(hashlib.sha256(raw).hexdigest()==args.request_sha256,'stopped driver request bytes differ')
    driver=StoppedCaptureDriver(role._decode(raw))
    print(json.dumps(driver.result()),flush=True)
    commands={'overlap','abort_overlap','retire','fence','capture','prepare','install','finish','status'}
    while not driver.finish_recorded:
        raw=sys.stdin.buffer.readline(65537)
        if not raw:
            print(json.dumps(driver.result(input_closed=True,custody_retained=True)),flush=True)
            while True:signal.pause()
        try:
            stopped.require(len(raw)<=65536,'stopped command exceeds bound')
            body=role._decode(raw);role._closed(body,{'command'})
            command=body['command'];stopped.require(command in commands,'unknown stopped driver command')
            value=driver.result() if command=='status' else getattr(driver,command)()
        except Exception as exc:
            value=driver.result(error=type(exc).__name__,reason=str(exc)[:256],custody_retained=True)
        print(json.dumps(value),flush=True)
    return 0
