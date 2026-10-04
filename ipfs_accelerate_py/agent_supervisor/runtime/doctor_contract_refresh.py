"""Invalidate observational local-contract proof reuse after publication.

The caller owns the completed-task / fenced-STOP gate. This helper reads no
owner state and grants no admission, proof, or completion authority. It only
observes the previously indexed source scope and retains old proof receipts
as historical evidence when their inputs change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

from ..proof.formal_verification_contracts import content_identity
from ..proof.proof_scope_index import (
    IndexedScopeRecord, ProofInputKind, ProofScopeBlobRecord, ProofScopeIndex,
    ProofScopeKey, update_proof_scope_index,
)
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .supervisor_meta_index import SupervisorMetaIndex


class DoctorContractRefreshError(ValueError):
    """A previous observation or its successor lost its exact input binding."""


MAX_FILE_BYTES = 16 * 1024 * 1024


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_under(root: Path, name: str, *, missing_allowed: bool) -> bytes | None:
    """Read inert bytes through no-follow directory descriptors, never imports."""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name or '..' in path.parts
            or any(part in {'.git', '.runtime'} for part in path.parts)):
        raise DoctorContractRefreshError('indexed source path is not a permitted relative path')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        for component in path.parts[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise DoctorContractRefreshError('indexed source is not a bounded regular file')
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise DoctorContractRefreshError('indexed source exceeded its read bound')
        return raw
    except FileNotFoundError:
        if missing_allowed:
            return None
        raise DoctorContractRefreshError('indexed artifact is missing') from None
    except OSError as exc:
        raise DoctorContractRefreshError('indexed source is unavailable or symlinked') from exc
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _capture(repository: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    result = {}
    for name in names:
        raw = _read_under(repository, name, missing_allowed=True)
        result[name] = None if raw is None else _sha(raw)
    return result


def refresh_doctor_contract_index(*, repository: Path, workflow: dict, state: Path) -> dict:
    """Rebuild native dependencies from a complete observation of the old scope.

    New files outside that scope are not analyzed. Unchanged receipts remain
    eligible only relative to these observed inputs; they are never promoted
    to a theorem about the current whole repository or the published program.
    """
    repository, state = Path(repository).absolute(), Path(state).absolute()
    if not repository.is_dir() or repository.resolve(strict=True) != repository:
        raise DoctorContractRefreshError('canonical repository root required')
    if workflow.get('repository') != str(repository):
        raise DoctorContractRefreshError('prior workflow identifies another repository')
    if state.resolve() != state or state.is_relative_to(repository) or state.exists():
        raise DoctorContractRefreshError('refresh state must be a new external directory')
    analysis = workflow.get('analysis')
    if not isinstance(analysis, dict) or analysis.get('analysis_cid') != content_identity({
            key: value for key, value in analysis.items() if key != 'analysis_cid'}):
        raise DoctorContractRefreshError('prior scoped analysis identity differs')
    ledger = analysis.get('source_hashes')
    if not isinstance(ledger, dict) or not ledger or any(
            type(name) is not str or type(digest) is not str or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)
            for name, digest in ledger.items()):
        raise DoctorContractRefreshError('prior scoped source ledger is malformed')
    catalog = workflow.get('contract_index', {})
    artifact = Path(catalog.get('artifact', '')).absolute()
    if artifact.resolve(strict=True) != artifact or artifact.is_relative_to(repository):
        raise DoctorContractRefreshError('prior proof index must be an exact external artifact')
    raw = _read_under(artifact.parent, artifact.name, missing_allowed=False)
    if _sha(raw) != catalog.get('artifact_sha256'):
        raise DoctorContractRefreshError('prior proof index digest differs')
    previous = ProofScopeIndex.from_dict(json.loads(raw))
    if previous.root_id != analysis.get('source_tree_id') or {
            blob.path: blob.blob_id for blob in previous.blobs} != ledger:
        raise DoctorContractRefreshError('prior proof index differs from its source ledger')
    if set(previous.active_receipt_ids) != set(catalog.get('active_receipt_ids', ())):
        raise DoctorContractRefreshError('prior active proof inventory differs')
    task_id = workflow.get('task_cid') or workflow.get('task_id')
    if not isinstance(task_id, str) or not task_id or task_id != analysis.get('task_cid'):
        raise DoctorContractRefreshError('prior workflow task identity differs from analysis')
    names = tuple(sorted(ledger))
    observed = _capture(repository, names)
    changed = tuple(name for name in names if observed[name] != ledger[name])
    deleted = tuple(name for name in names if observed[name] is None)
    next_root = content_identity({'schema': 'doctor-observed-scoped-source-root@1',
        'parent_source_tree_id': previous.root_id, 'analysis_cid': analysis['analysis_cid'],
        'source_hashes': observed})
    blobs = []
    for name, digest in observed.items():
        if digest is None:
            continue
        key = ProofScopeKey(ProofInputKind.FILE, name)
        scope = IndexedScopeRecord(content_identity({'path': name, 'sha256': digest}), name, digest, (key,))
        blobs.append(ProofScopeBlobRecord(name, digest, (scope,)))
    index = update_proof_scope_index(previous, scope_blobs=blobs,
        obligations=previous.obligations, receipts=previous.receipts, root_id=next_root,
        changed_inputs=tuple(ProofScopeKey(ProofInputKind.FILE, name) for name in changed))
    if not set(index.active_receipt_ids) <= set(previous.active_receipt_ids):
        raise DoctorContractRefreshError('refresh attempted to promote a proof receipt')
    if _capture(repository, names) != observed:
        raise DoctorContractRefreshError('scoped source changed during index reconstruction')
    state.mkdir(parents=True, mode=0o700)
    index_body = index.to_dict()
    successor = state / 'proof-scope-index.json'
    successor.write_text(json.dumps(index_body, sort_keys=True, indent=2) + '\n')
    record = {'schema': 'supervisor-doctor-contract-refresh@1', 'task_id': task_id,
        'board': 'doctor-contracts', 'operation': 'published_scoped_contract_invalidation',
        'analysis_cid': analysis['analysis_cid'], 'parent_index_id': previous.index_id,
        'parent_artifact_sha256': catalog['artifact_sha256'], 'parent_source_tree_id': previous.root_id,
        'observed_scope_root': next_root, 'source_hashes': observed,
        'changed_paths': list(changed), 'deleted_paths': list(deleted),
        'proof_index': index_body, 'active_receipt_ids': list(index.active_receipt_ids),
        'invalidated_receipt_ids': list(index.invalidated_receipt_ids),
        'historical_proof_receipt_ids': [receipt.receipt_id for receipt in previous.receipts],
        'new_proof_receipts': 0, 'provider_calls': 0,
        'whole_program_proved': False, 'whole_repository_freshness_checked': False,
        'scope_expanded': False, 'completion_authority': False, 'publication_authority': False,
        'freshness_scope': 'previously indexed source paths only', 'proposal_only': True}
    world = ProgramWorldDatabase(state / 'contracts.duckdb', state / 'contracts-lake')
    persisted = world.persist(record)
    hydrated = world.records_for_decision(task_id=task_id, operation=record['operation'])
    if hydrated['n'] != 1 or hydrated['records'][0]['payload']['proof_index'] != index_body:
        raise DoctorContractRefreshError('successor native world hydration differs')
    metadata = SupervisorMetaIndex(state / 'metadata.duckdb', state / 'metadata-lake')
    successor_sha = _sha(successor.read_bytes())
    for kind, locator, ref in (
        ('world_model', state / 'contracts.duckdb', persisted['record_cid']),
        ('proof_certificate', successor, content_identity({'artifact_sha256': successor_sha})),
    ):
        registered = metadata.register_catalog(kind=kind, locator_ref=str(locator),
            tree_id=next_root, project=False)
        metadata.link_identity(subject_kind='task_id', subject_ref=task_id,
            catalog_id=registered['catalog_id'], record_kind=kind, record_ref=ref, project=False)
    projection = metadata.project_ducklake()
    if _capture(repository, names) != observed:
        raise DoctorContractRefreshError('scoped source changed during successor persistence')
    result = {**record, 'status': 'observed', 'world_record': persisted, 'metadata': projection,
        'artifact': str(successor), 'artifact_sha256': successor_sha, 'hydrated': True}
    (state / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result
