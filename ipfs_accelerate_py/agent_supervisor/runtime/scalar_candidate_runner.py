"""Materialize owner-selected scalar bytes in the allocated native worktree.

The owner pins the artifact digest in its independently admitted launch. This
worker checks that binding and exact source preimages, and neither evaluates
source nor publishes commits or completes tasks. Retained bounded proof digests
are evidence identifiers, not live proof or execution capabilities.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import stat
import sys

from ..proof.formal_verification_contracts import content_identity
from .doctor_candidate_runner import MAX_BYTES, _directory, _git, _read, _sha, _unique
from .doctor_contract_candidate_runner import _check_parent, _path, _write
from .scalar_candidate_handoff import SCHEMA, SCOPE, FALSE

FIELDS = {'schema','repository','baseline_commit','task_cid','task_id','task_revision','manifest_cid',
    'instruction_binding','permitted_output','edit','evidence','scope','provider_calls','artifact_cid',*FALSE}


def _digest(value):
    return type(value) is str and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def materialize_scalar_candidate(*, artifact: Path, expected_sha256: str, task_cid: str,
                                 prompt: str, workspace: Path) -> dict:
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, info = _read(parent, artifact.name)
    finally:
        os.close(parent)
    if stat.S_IMODE(info.st_mode) & 0o222 or not _digest(expected_sha256) or _sha(raw) != expected_sha256:
        raise ValueError('immutable owner-pinned scalar handoff required')
    payload = json.loads(raw, object_pairs_hook=_unique)
    if type(payload) is not dict or set(payload) != FIELDS or payload['schema'] != SCHEMA:
        raise ValueError('closed scalar handoff schema required')
    if (payload['artifact_cid'] != content_identity({k:v for k,v in payload.items() if k != 'artifact_cid'})
            or payload['task_cid'] != task_cid or type(payload['task_revision']) is not int
            or payload['task_revision'] < 1 or type(payload['provider_calls']) is not int
            or payload['provider_calls'] != 0 or payload['scope'] != SCOPE
            or any(payload[name] is not False for name in FALSE)
            or any(type(payload[name]) is not str or not payload[name] or len(payload[name]) > 8192
                   for name in ['task_cid','task_id','manifest_cid'])):
        raise ValueError('scalar handoff identity or authority differs')
    binding=payload['instruction_binding']
    if (type(binding) is not dict or set(binding) != {'kind','source_path','instruction_sha256'}
            or binding['kind'] not in {'signed_task_objective','signed_source'} or not _digest(binding['instruction_sha256'])
            or (binding['kind']=='signed_task_objective' and binding['source_path'] is not None)):
        raise ValueError('closed signed instruction binding required')
    if binding['kind']=='signed_source': _path(binding['source_path'])
    evidence=payload['evidence']
    if (type(evidence) is not dict or set(evidence) != {'repair_advice_sha256','candidate_id','candidate_sha256',
            'enabled_case_count','input_sha256','scalar_producer_sha256'}
            or any(not _digest(evidence[key]) for key in ['repair_advice_sha256','candidate_sha256','input_sha256','scalar_producer_sha256'])
            or type(evidence['enabled_case_count']) is not int or not 1 <= evidence['enabled_case_count'] <= 64
            or type(evidence['candidate_id']) is not str or not evidence['candidate_id']):
        raise ValueError('bounded scalar evidence identifiers required')
    if type(prompt) is not str or len(prompt.encode()) > 256000:
        raise ValueError('bounded native prompt required')
    wire,_=json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if type(wire) is not dict or wire.get('objective_id') != payload['task_id']:
        raise ValueError('native task differs from scalar handoff')
    root=Path(workspace).absolute();canonical=Path(payload['repository'])
    if (root.resolve(strict=True)!=root or not canonical.is_absolute() or canonical.resolve(strict=True)!=canonical
            or root==canonical or artifact.is_relative_to(root)):
        raise ValueError('separate native allocated worktree and owner artifact required')
    expected_artifact = canonical / '.runtime/scalar-handoffs' / (expected_sha256 + '.json')
    owner_uid = canonical.stat().st_uid
    if artifact != expected_artifact:
        raise ValueError('scalar artifact is outside its exact owner-controlled location')
    for location in (canonical / '.runtime', expected_artifact.parent, expected_artifact):
        metadata = location.lstat()
        expected_kind = stat.S_ISREG if location == expected_artifact else stat.S_ISDIR
        if metadata.st_uid != owner_uid or not expected_kind(metadata.st_mode) or metadata.st_mode & 0o022:
            raise ValueError('scalar artifact or directory is not owner-controlled')
    if (Path(_git(root,'rev-parse','--show-toplevel').decode().strip())!=root
            or _git(root,'rev-parse','--path-format=absolute','--git-common-dir') !=
                _git(canonical,'rev-parse','--path-format=absolute','--git-common-dir')):
        raise ValueError('scalar worktree belongs to a foreign repository')
    baseline=payload['baseline_commit']
    if (type(baseline) is not str or re.fullmatch(r'[a-f0-9]{40}',baseline) is None
            or _git(root,'rev-parse','HEAD').decode().strip()!=baseline
            or _git(canonical,'rev-parse','HEAD').decode().strip()!=baseline):
        raise ValueError('scalar baseline drifted')
    output=payload['permitted_output'];edit=payload['edit']
    if (type(output) is not dict or set(output)!={'path','effect','media_type'} or output['effect']!='modify'
            or type(output['media_type']) is not str or not output['media_type']
            or type(edit) is not dict or set(edit)!={'path','before_sha256','after_sha256','after_bytes_base64'}
            or output['path']!=edit['path'] or not _digest(edit['before_sha256']) or not _digest(edit['after_sha256'])):
        raise ValueError('single declared scalar modification required')
    relative=_path(edit['path'])
    after=base64.b64decode(edit['after_bytes_base64'],validate=True)
    if len(after)>MAX_BYTES or _sha(after)!=edit['after_sha256']:
        raise ValueError('scalar candidate bytes differ')
    original=_git(root,'show',baseline+':'+str(relative))
    if _sha(original)!=edit['before_sha256'] or after==original:
        raise ValueError('scalar candidate preimage differs')
    canonical_parent = _directory(canonical / relative.parent)
    try:
        if _read(canonical_parent, relative.name)[0] != original:
            raise ValueError('canonical scalar preimage drifted')
    finally:
        os.close(canonical_parent)
    if binding['kind'] == 'signed_source':
        instruction_relative = _path(binding['source_path'])
        instruction_original = _git(root, 'show', baseline + ':' + str(instruction_relative))
        instruction_parent = _directory(root / instruction_relative.parent)
        try:
            if (_sha(instruction_original) != binding['instruction_sha256']
                    or _read(instruction_parent, instruction_relative.name)[0] != instruction_original):
                raise ValueError('allocated signed instruction drifted')
        finally:
            os.close(instruction_parent)
    parent=_directory(root/relative.parent)
    try:
        current,metadata=_read(parent,relative.name)
        if current!=original:raise ValueError('allocated scalar preimage drifted')
        _check_parent(root,relative,parent)
        mode=_write(parent,relative,after,metadata)
        _check_parent(root,relative,parent)
        if _read(parent,relative.name)[0]!=after or _git(root,'rev-parse','HEAD').decode().strip()!=baseline:
            raise ValueError('scalar candidate changed during materialization')
    finally:os.close(parent)
    return dict(schema='native-owner-scalar-candidate-materialization@1',status='candidate_materialized',
        artifact_cid=payload['artifact_cid'],task_cid=task_cid,baseline_commit=baseline,
        changed_paths=[str(relative)],source_after_sha256=edit['after_sha256'],write_mode=mode,
        evidence=payload['evidence'],provider_calls=0,**FALSE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',required=True,type=Path)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--task-cid',required=True)
    args=parser.parse_args()
    try:
        result=materialize_scalar_candidate(artifact=args.artifact,expected_sha256=args.sha256,
            task_cid=args.task_cid,prompt=sys.stdin.buffer.read(256001).decode(),workspace=Path.cwd())
    except Exception as error:
        print(json.dumps(dict(schema='native-owner-scalar-candidate-materialization@1',status='refused',
            error_type=type(error).__name__,completion_authority=False)),file=sys.stderr)
        return 1
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
