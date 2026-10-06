"""Publish reviewed exact files through the native append-only Hub owner.

Requires an explicit preparation digest and --publish. Existing repositories
only; all prior path identities, visibility and original asset pins are checked.
No ModelManager record, runtime binding, cache, encoder or model is changed.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import os
from pathlib import Path
import stat
import sys

OUT = Path(__file__).resolve().parent
FORBIDDEN = ('torch', 'transformers', 'numpy', 'sentence_transformers', 'duckdb',
             'ipfs_accelerate_py.model_manager')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def witness(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def capture(path, expected=None):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 16 * 1024 * 1024,
            'bounded independent regular pinned file required')
    sha, count = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        require(witness(os.fstat(descriptor)) == witness(before), 'pinned file changed before open')
        while True:
            block = os.read(descriptor, min(1024 * 1024, before.st_size - count + 1))
            if not block:
                break
            count += len(block)
            require(count <= before.st_size, 'pinned file grew')
            sha.update(block)
        require(count == before.st_size and witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()),
                'pinned file changed during read')
    finally:
        os.close(descriptor)
    pin = {'path': str(path), 'bytes': count, 'sha256': sha.hexdigest()}
    require(expected is None or pin == expected, 'reviewed pinned bytes differ')
    return pin


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return capture(path)


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            raise ImportError('model/database import forbidden during publication')
        return None


def file_identity(row):
    lfs = row.lfs
    return {'path': row.rfilename, 'bytes': row.size, 'blob_id': row.blob_id,
        'lfs': None if lfs is None else {'sha256': lfs.sha256, 'bytes': lfs.size}}


def identities(info):
    rows = {row.rfilename: file_identity(row) for row in info.siblings}
    require(len(rows) == len(info.siblings), 'ambiguous remote path metadata')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation-sha256', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    preparation_pin = capture(OUT / 'preparation.json')
    require(preparation_pin['sha256'] == args.preparation_sha256, 'preparation differs from reviewed digest')
    preparation = json.loads((OUT / 'preparation.json').read_text())
    require(preparation['completed'] is True and preparation['planned_repository_count'] == 2,
            'reviewed two-repository preparation required')
    originals = preparation['all_original_and_staged_file_pins']
    for pin in originals:
        capture(pin['path'], pin)
    owner_pin = preparation['native_publisher_source']['file_pin']
    capture(owner_pin['path'], owner_pin)
    spec = importlib.util.spec_from_file_location('dimension_mirror_native_publish', owner_pin['path'])
    native = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, NoModels())
    spec.loader.exec_module(native)
    plans = []
    for lane in preparation['lanes']:
        plan_pin = lane['publication_plan_pin']
        capture(plan_pin['path'], plan_pin)
        plan = json.loads(Path(plan_pin['path']).read_text())
        require(plan['repository_id'] == lane['repository_id']
                and plan['repository_id'] in ('Publicus/legal-ir-autoencoder-384d', 'Publicus/legal-ir-autoencoder-768d')
                and plan['private_new'] is False and len(plan['operations']) == 3,
                'exact existing public repository and three-file plan required')
        require(all(row['path_in_repo'].startswith(preparation['release_prefix'] + '/')
                    for row in plan['operations']), 'only new release subtree is authorized')
        native._freeze_files(native._capture_plan(plan))
        plans.append((lane, plan))
    if not args.publish:
        print(json.dumps({'status': 'local_preflight_only', 'preparation_pin': preparation_pin,
                          'remote_calls': 0, 'Hub_mutations': 0}))
        return

    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    from huggingface_hub import HfApi
    api = HfApi()
    outcomes = []
    for lane, plan in plans:
        repo = lane['repository_id']
        width = lane['dimension']
        before = api.model_info(repo, files_metadata=True)
        require(before.id == repo and before.sha == lane['observed_initial_revision']
                and before.private is False, 'existing public repository changed since reviewed preparation')
        baseline = identities(before)
        require(len(baseline) == lane['observed_initial_file_count'], 'baseline file population changed')
        require(not any(row['path_in_repo'] in baseline for row in plan['operations']),
                'reviewed new subtree is no longer empty')
        baseline_pin = write_new(OUT / (str(width) + 'd-before-publication.json'),
            {'repository_id': repo, 'revision': before.sha, 'private': before.private,
             'gated': before.gated, 'files': baseline})

        class ExistingOnlyAPI:
            def model_info(self, repository_id, **kwargs):
                require(repository_id == repo, 'unreviewed repository request')
                return api.model_info(repository_id, **kwargs)
            def create_repo(self, *args, **kwargs):
                raise ValueError('repository creation forbidden; existing repositories only')
            def create_commit(self, repository_id, **kwargs):
                require(repository_id == repo and kwargs['parent_commit'] == before.sha,
                        'remote parent differs from independently observed baseline')
                require({operation.path_in_repo for operation in kwargs['operations']} ==
                        {row['path_in_repo'] for row in plan['operations']}, 'only explicit additions are authorized')
                return api.create_commit(repository_id, **kwargs)

        try:
            receipt = native.publish_ir_model_hub_release(plan, api=ExistingOnlyAPI())
        except native.HubPublicationError as error:
            failure_pin = write_new(OUT / (str(width) + 'd-publication-failure.json'),
                {'schema': 'dimension-mirror-publication-failure/v1', 'error_type': type(error).__name__,
                 'reason': str(error), 'partial_receipt': error.receipt,
                 'baseline_pin': baseline_pin, 'runtime_admitted': False})
            print(json.dumps({'status': 'publication_refused', 'failure_pin': failure_pin}))
            return
        receipt_pin = write_new(OUT / (str(width) + 'd-publication-receipt.json'), receipt)
        after = api.model_info(repo, revision=receipt['revision'], files_metadata=True)
        current = identities(after)
        require(after.private == before.private and after.gated == before.gated
                and all(current.get(path) == value for path, value in baseline.items()),
                'preexisting path identities or repository visibility changed')
        require(set(current) - set(baseline) == {row['path_in_repo'] for row in plan['operations']},
                'unexpected added repository paths')
        for pin in originals:
            capture(pin['path'], pin)
        capture(OUT / 'preparation.json', preparation_pin)
        preservation = {'schema': 'dimension-mirror-prior-path-preservation/v1',
            'repository_id': repo, 'initial_revision': before.sha, 'published_revision': receipt['revision'],
            'initial_file_count': len(baseline), 'published_file_count': len(current),
            'all_prior_path_identities_preserved': True, 'visibility_preserved': True,
            'gated_setting_preserved': True, 'original_and_staged_file_pins_preserved': True,
            'new_paths': sorted(set(current) - set(baseline)), 'baseline_pin': baseline_pin,
            'publication_receipt_pin': receipt_pin, 'root_defaults_and_model_card_untouched': True,
            'original_model_manager_record_rewritten': False, 'model_loaded': False,
            'runtime_admitted': False, 'proof_authority': False}
        preservation_pin = write_new(OUT / (str(width) + 'd-preservation.json'), preservation)
        outcomes.append({'dimension': width, 'repository_id': repo, 'revision': receipt['revision'],
                         'receipt_pin': receipt_pin, 'preservation_pin': preservation_pin})
    summary = {'schema': 'retained-contextual-dimension-mirror-publication/v1', 'completed': True,
        'preparation_pin': preparation_pin, 'outcomes': outcomes,
        'new_repository_count': 0, 'new_release_files': 6, 'models_changed_or_retrained': False,
        'original_model_manager_records_rewritten': False, 'runtime_admitted': False,
        'proof_authority': False, 'cross_repository_transaction_atomic': False}
    pin = write_new(OUT / 'publication-summary.json', summary)
    print(json.dumps({'status': 'published_verified', 'publication_summary_pin': pin, 'outcomes': outcomes}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Raw SDK exception strings can contain request details; keep them out
        # of human-facing output and preserve only this stable failure type.
        print(json.dumps({'status': 'refused', 'error_type': type(error).__name__}))
        sys.exit(1)
