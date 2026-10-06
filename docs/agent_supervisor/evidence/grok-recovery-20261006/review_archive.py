"""Verify frozen source inventory and actual pinned Grok member, without models."""
from pathlib import PurePosixPath
import argparse
import hashlib
import json
import re
import os
import stat
import tarfile
import sys
import time
import run_fresh_grok as recipe

MAX_ARCHIVE_BYTES = 16 * 1024**3
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_MEMBER_BYTES = 16 * 1024**3
MAX_MEMBERS = 100_000


def inventory_by_name(files):
    if type(files) is not list or not 0 < len(files) <= MAX_MEMBERS:
        raise ValueError('bounded nonempty archive inventory required')
    result = {}
    total = 0
    for row in files:
        if type(row) is not dict:
            raise ValueError('archive inventory row must be an object')
        name = row.get('path')
        if (type(name) is not str or not 0 < len(name) <= 4096 or '\\' in name
                or name.startswith('/') or any(part in {'', '.', '..', '.env', 'auth.json'} for part in name.split('/'))
                or name in result):
            raise ValueError('canonical unique noncredential archive path required')
        if (type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= MAX_MEMBER_BYTES
                or type(row.get('sha256')) is not str or re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is None
                or type(row.get('mode')) is not int or row['mode'] not in {0o644, 0o755}):
            raise ValueError('bounded pinned regular archive inventory required')
        total += row['bytes']
        if total > MAX_TOTAL_MEMBER_BYTES:
            raise ValueError('archive expansion exceeds its byte bound')
        result[name] = row
    return result


def verify_archive_content(archive, manifest):
    """Stream actual archive and every declared member; never extract paths."""
    inventory = inventory_by_name(manifest.get('files'))
    expected = manifest.get('archive_sha256')
    if type(expected) is not str or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
        raise ValueError('independent archive digest required')
    if archive.is_symlink() or archive.resolve(strict=True) != archive.absolute():
        raise ValueError('canonical regular archive required')
    fd = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_ARCHIVE_BYTES:
            raise ValueError('bounded regular archive required')
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('actual archive digest differs from manifest')
        stream.seek(0)
        seen = set()
        counts = dict.fromkeys(('source', 'datasets', 'kit'), 0)
        total = 0
        with tarfile.open(fileobj=stream, mode='r|gz') as archive_stream:
            for member in archive_stream:
                name = member.name
                row = inventory.get(name)
                if row is None or name in seen or len(seen) >= MAX_MEMBERS:
                    raise ValueError('archive member is absent from inventory or duplicated')
                if (not member.isfile() or member.issparse() or member.size != row['bytes']
                        or member.mode != row['mode'] or member.linkname
                        or member.uid != 0 or member.gid != 0):
                    raise ValueError('archive member differs from pinned regular inventory')
                seen.add(name)
                body = archive_stream.extractfile(member)
                if body is None:
                    raise ValueError('archive member body unavailable')
                digest = hashlib.sha256()
                count = 0
                with body:
                    while block := body.read(1024 * 1024):
                        count += len(block)
                        if count > row['bytes']:
                            raise ValueError('archive member exceeds declared length')
                        digest.update(block)
                if count != row['bytes'] or digest.hexdigest() != row['sha256']:
                    raise ValueError('actual archive member bytes differ from manifest')
                total += count
                prefix = name.partition('/')[0]
                if prefix in counts:
                    counts[prefix] += 1
        if seen != set(inventory):
            raise ValueError('archive is missing declared inventory members')
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(archive.stat()):
        raise ValueError('archive changed during independent review')
    return dict(archive_sha256_actual=actual, archive_hash_recomputed_in_review=True,
        archive_member_bytes_verified=True, archive_verified_member_count=len(seen),
        archive_verified_uncompressed_bytes=total, archive_source_verified_files=counts,
        archive_member_verification='All declared members have matching bytes, lengths, regular types and modes; none extracted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-source-head', required=True)
    parser.add_argument('--bundle-attempt', default='01')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.expected_source_head) or not re.fullmatch(r'[0-9]{2}', args.bundle_attempt):
        parser.error('full committed source SHA and two-digit attempt required')
    destination = recipe.OUTPUT / ('grok-container/archive-review-' + args.bundle_attempt + '.json')
    if destination.exists():
        raise SystemExit('fresh archive review required')
    started = time.monotonic()
    before = recipe.snapshot(args.expected_source_head)
    stem = 'grok-bundle-' + args.bundle_attempt
    bundle = recipe.OUTPUT / stem
    command = json.loads((recipe.OUTPUT / (stem + '-command.json')).read_text())
    exit_record = json.loads((recipe.OUTPUT / (stem + '-exit.json')).read_text())
    assert command['source_heads'] == before['source_heads'] and not any(command['source_status'].values())
    assert exit_record['exit_code'] == 0 and exit_record['source_after'] == before
    raw = (bundle / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    assert manifest['credentials_in_archive'] is False and manifest['task_inputs_in_archive'] is False
    assert 'setup_cache' not in manifest
    roots = {'source': recipe.SOURCE, 'datasets': recipe.DATASETS, 'kit': recipe.KIT}
    archive_review = verify_archive_content(bundle / 'runtime.tar.gz', manifest)
    seen, counts = set(), dict.fromkeys(roots, 0)
    for row in manifest['files']:
        name = row['path']
        parts = PurePosixPath(name).parts
        assert name not in seen and not name.startswith('/') and '..' not in parts
        seen.add(name)
        assert not any(part in {'.env', 'auth.json'} for part in parts)
        prefix, _, relative = name.partition('/')
        if prefix in roots:
            path = roots[prefix] / relative
            assert path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
            counts[prefix] += 1
    sys.path.insert(0, str(recipe.SOURCE))
    from benchmarks.agent_supervisor.container_coding.terminal_grok_deployment import verify_grok_archive
    binding = verify_grok_archive(bundle / 'runtime.tar.gz', manifest, required=True)
    assert recipe.snapshot(args.expected_source_head) == before
    result = dict(schema='terminal-grok-archive-independent-review@1', qualified=True,
        archive_bytes=(bundle / 'runtime.tar.gz').stat().st_size,
        archive_sha256_declared=manifest['archive_sha256'], **archive_review,
        manifest_sha256=hashlib.sha256(raw).hexdigest(), inventory_files=len(manifest['files']),
        source_inventory_verified_files=counts,
        source_inventory_comparison='Every declared source/datasets/kit member matches its current clean pinned checkout.',
        grok_archive_member_independently_verified=binding, credentials_in_archive=False,
        credential_pathnames_present=False, task_inputs_in_archive=False, codex_setup_cache_selected=False,
        provider_calls=0, docker_calls=0, raw_file_bodies_exported=False, benchmark_result=False,
        seconds=time.monotonic() - started, **before)
    recipe.write_json(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
