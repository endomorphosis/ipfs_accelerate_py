"""Owner-selected bounded scalar candidates for an independently admitted task.

Fresh learned inference and Lake results select candidate bytes. They do not
supply task, execution, publication, or completion authority. The existing
signed owner admission supplies the task scope; the native daemon still runs
its validation and publication gates. This is not Doctor composition.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
import stat
from pathlib import Path

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import IntentRepository
from . import local_planning_admission as local
from .doctor_candidate_runner import MAX_BYTES, _directory, _read
from .doctor_contract_candidate_runner import _path

SCHEMA = 'supervisor-owner-scalar-candidate@1'
RESULT_SCHEMA = 'supervisor-owner-scalar-candidate-preparation@1'
SCOPE = 'explicit finite input domains and source-audited scalar operator candidate; native task validation and publication remain required'
FALSE = dict(proof_authority=False, execution_authority=False, mutation_authority=False,
    publication_authority=False, completion_authority=False, doctor_composition_used=False,
    whole_program_semantics_verified=False, source_semantics_verified=False)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _source(repository, path):
    relative = _path(path)
    parent = _directory(repository / relative.parent)
    try:
        raw, _ = _read(parent, relative.name)
        return raw
    finally:
        os.close(parent)


def _persist(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())
    parent = _directory(path.parent)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)



def _public_handoff(repository, raw):
    """Expose only inert pinned candidate bytes through owner-controlled paths."""
    directory = repository
    for component in ('.runtime', 'scalar-handoffs'):
        directory = directory / component
        try:
            directory.mkdir(mode=0o755)
        except FileExistsError:
            pass
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022
                or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError('scalar handoff directories must be owner-controlled and worker-readable')
    artifact = directory / (_sha(raw) + '.json')
    _persist(artifact, raw)
    return artifact

def prepare_scalar_candidate_handoff(*, repository: Path, admission: dict,
        intent: IntentRepository, task_cid: str, state: Path, instruction: str,
        intent_config: dict, security_config: dict, effect_config: dict,
        instruction_path: str | None = None, maximum_candidates: int = 2) -> dict:
    """Call fresh scalar inference/proofs; export exactly one admitted-scope edit.

    The instruction is either the exact signed task objective or an immutable
    signed source file. There is no saved-advice input or proof-boolean override.
    All unsuccessful candidate populations remain in the retained advice.
    """
    from . import scalar_repair_advisor as scalar
    if not isinstance(intent, IntentRepository):
        raise ValueError('native IntentRepository required')
    repository, state = Path(repository).absolute(), Path(state).absolute()
    if repository.resolve(strict=True) != repository or not repository.is_dir():
        raise ValueError('exact canonical owner repository required')
    if state.resolve() != state or state.exists() or state.is_relative_to(repository):
        raise ValueError('fresh external scalar candidate state required')
    if type(instruction) is not str or not instruction.strip() or len(instruction.encode()) > 65536:
        raise ValueError('bounded original instruction required')
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 2:
        raise ValueError('one or two bounded scalar candidates required')
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest = verified['manifest']
    if manifest['repository'] != str(repository):
        raise ValueError('scalar candidate repository differs from signed admission')
    task = intent.get_task(task_cid)
    native = [row for row in verified['graph'].tasks if row.task_cid == task_cid]
    if task is None or len(native) != 1 or task['status'] not in {'ready', 'in_progress'}:
        raise ValueError('current active independently admitted task required')
    contract, _, _, _ = local._contract(task['body'], task_cid)
    if (contract['manifest_cid'] != verified['receipt']['manifest_cid']
            or task['task_alias'] != native[0].task_key or task['goal_cid'] != native[0].goal_cid):
        raise ValueError('scalar candidate task differs from signed graph')
    outputs = contract['task_spec']['outputs']
    if len(outputs) != 1 or outputs[0]['effect'] != 'modify':
        raise ValueError('scalar candidate requires exactly one declared modify output')
    path = outputs[0]['path']
    _path(path)
    if not path.endswith('.py') or path not in manifest['sources']:
        raise ValueError('scalar candidate requires an admitted Python source')
    raw = _source(repository, path)
    source_hash = _sha(raw)
    if source_hash != manifest['sources'][path]['sha256']:
        raise ValueError('scalar source differs from signed preimage')
    instruction_hash = _sha(instruction.encode())
    if instruction_path is None:
        if instruction != native[0].objective:
            raise ValueError('instruction must equal the signed task objective')
        instruction_binding = {'kind': 'signed_task_objective', 'source_path': None,
            'instruction_sha256': instruction_hash}
    else:
        _path(instruction_path)
        if instruction_path == path or instruction_path not in manifest['sources']:
            raise ValueError('instruction must be an immutable signed non-output source')
        instruction_raw = _source(repository, instruction_path)
        if instruction_raw != instruction.encode() or _sha(instruction_raw) != manifest['sources'][instruction_path]['sha256']:
            raise ValueError('instruction differs from signed source bytes')
        instruction_binding = {'kind': 'signed_source', 'source_path': instruction_path,
            'instruction_sha256': instruction_hash}
    options = dict(instruction=instruction, intent_config=intent_config, security_config=security_config,
        effect_config=effect_config, maximum_candidates=maximum_candidates,
        source_rows=[dict(id=path, source_text=raw.decode('utf-8'), source_sha256=source_hash)])
    original_options = _wire(options)
    source_pin = _sha(Path(scalar.__file__).read_bytes())

    def current():
        local.verify_local_benchmark_admission(admission, initial=True)
        if (intent.get_task(task_cid) != task or _source(repository, path) != raw
                or _wire(options) != original_options
                or _sha(Path(scalar.__file__).read_bytes()) != source_pin):
            raise ValueError('source, task, inference inputs or scalar producer changed')
        if instruction_path is not None and _source(repository, instruction_path) != instruction.encode():
            raise ValueError('signed instruction changed during candidate preparation')

    current()
    advice = scalar.prepare_scalar_repair_advice(**deepcopy(options))
    current()
    advice_raw = _wire(advice)
    if len(advice_raw) > 16 * 1024 * 1024:
        raise ValueError('scalar repair evidence exceeds retained byte bound')
    result = dict(schema=RESULT_SCHEMA, status='residual', reason_codes=[], task_cid=task_cid,
        task_id=task['task_alias'], task_revision=task['revision'], source_path=path,
        instruction_binding=instruction_binding, repair_advice=advice,
        repair_advice_sha256=_sha(advice_raw), handoff_path=None, handoff_sha256=None,
        handoff=None, provider_calls=0, scope=SCOPE, **FALSE)
    if (type(advice) is not dict or advice.get('schema') != 'supervisor-scalar-operator-repair-advice/v1'
            or any(advice.get(key) is not False for key in ('proof_authority','execution_authority',
                'mutation_authority','completion_authority','source_semantics_verified','repair_applied'))):
        raise ValueError('fresh scalar advisor returned incompatible evidence or authority')
    candidates = advice.get('candidates', [])
    if type(candidates) is not list or len(candidates) > 2 or any(type(row) is not dict for row in candidates):
        raise ValueError('bounded complete scalar candidate population required')
    accepted = [row for row in candidates if row.get('status') == 'checked_candidate'
        and row.get('effect_status') == 'satisfied' and row.get('live_build_verified') is True
        and row.get('bounded_effects_satisfied') is True and type(row.get('enabled_case_count')) is int
        and 1 <= row['enabled_case_count'] <= 64]
    if (advice.get('status') != 'candidate_evidence' or advice.get('initial_refutation_live_verified') is not True
            or advice.get('input_pins_rechecked') is not True
            or advice.get('original_source_unchanged') is not True):
        result['reason_codes'] = ['fresh_refutation_and_candidate_evidence_unavailable']
    elif len(candidates) != 2 or any(row.get('status') != 'checked_candidate'
            or row.get('live_build_verified') is not True for row in candidates):
        result['reason_codes'] = ['candidate_population_incomplete']
    elif len(accepted) != 1:
        result['reason_codes'] = ['ambiguous_satisfied_candidates' if len(accepted) > 1 else 'no_nonvacuous_satisfied_candidate']
    else:
        if advice.get('original_source') != options['source_rows'][0]:
            raise ValueError('fresh scalar advice source differs from signed source')
        chosen = accepted[0]
        if advice.get('satisfied_candidate_ids') != [chosen['id']]:
            raise ValueError('scalar satisfied candidate population differs')
        after = chosen['source_text'].encode('utf-8')
        if (chosen['before_source_sha256'] != source_hash or chosen['source_sha256'] != _sha(after)
                or len(after) > MAX_BYTES or after == raw):
            raise ValueError('selected scalar candidate source identity differs')
        payload = dict(schema=SCHEMA, repository=str(repository), baseline_commit=manifest['baseline_commit'],
            task_cid=task_cid, task_id=task['task_alias'], task_revision=task['revision'],
            manifest_cid=contract['manifest_cid'], instruction_binding=instruction_binding,
            permitted_output=deepcopy(outputs[0]),
            edit=dict(path=path, before_sha256=source_hash, after_sha256=_sha(after),
                after_bytes_base64=base64.b64encode(after).decode()),
            evidence=dict(repair_advice_sha256=_sha(advice_raw), candidate_id=chosen['id'],
                candidate_sha256=_sha(_wire(chosen)), enabled_case_count=chosen['enabled_case_count'],
                input_sha256=_sha(original_options), scalar_producer_sha256=source_pin),
            scope=SCOPE, provider_calls=0, **FALSE)
        payload['artifact_cid'] = content_identity(payload)
        encoded = _wire(payload)
        if len(encoded) > MAX_BYTES:
            raise ValueError('scalar candidate handoff exceeds worker bound')
        current()
        result.update(status='candidate_ready', handoff=payload, handoff_sha256=_sha(encoded),
            handoff_path=None)
    state.mkdir(mode=0o700, parents=True)
    _persist(state/'repair-advice.json', advice_raw)
    if result['handoff'] is not None:
        result['handoff_path'] = str(_public_handoff(repository, _wire(result['handoff'])))
    _persist(state/'result.json', _wire(result))
    current()
    return result
