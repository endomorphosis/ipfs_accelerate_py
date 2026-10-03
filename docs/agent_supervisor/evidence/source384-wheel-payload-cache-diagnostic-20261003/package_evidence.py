"""Preserve the completed top128 diagnostic, excluding all task/model bodies."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
OUT = BASE / 'public-evidence'
NAMES = (
    'verdict.json', 'resource-analysis.json', 'native-resource-events.jsonl',
    'native-resource-collection.json', 'post-context-scheduler.json',
    'controls-03.json', 'controls-03.log', 'independent-review.json',
    'independent-row-audit.json', 'pip-mode-review-binding.json',
    'source-pin-reproduction.json', 'finite-library-source-pins.json',
    'final-sources.json', 'before-pins.json', 'after-pins.json',
    'scope.json', 'experiment-provenance.json', 'host-archive-provenance.json',
    'run-command.json', 'launch.json', 'launcher-exit.json', 'exit.json',
    'failure.json', 'containers-before.json', 'containers-after.json',
    'cache-advice.json', 'cache-advice-status.json',
    'codex-cache-advice.json', 'codex-cache-advice-status.json',
    'library-cache-advice.json', 'library-cache-advice-status.json',
    'cache_candidate.py', 'codex_cache_candidate.py',
    'native_library_cache_candidate.py', 'pin_native_libraries.py',
    'run_diagnostic.py', 'resource_runtime.py',
    'test_codex_cache_candidate.py', 'test_native_library_cache_candidate.py',
    'test_resource_runtime.py',
    'docker-01/source384-context.json', 'docker-01/source384-probe.stdout',
    'docker-01/source384-probe.stderr', 'docker-01/source384-probe-status.json',
    'docker-01/cache-resources-before.json', 'docker-01/cache-resources-after.json',
    'docker-01/codex-cache-resources-before.json', 'docker-01/codex-cache-resources-after.json',
    'docker-01/library-cache-resources-before.json', 'docker-01/library-cache-resources-after.json',
    'docker-01/resources.json', 'docker-01/detailed-resources.json',
    'docker-01/admission-estimate.json', 'package_evidence.py',
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def main():
    if OUT.exists():
        raise ValueError('fresh immutable package required')
    verdict = json.loads((BASE / 'verdict.json').read_text())
    for name, digest in verdict['files'].items():
        assert sha(BASE / name) == digest, name
    assert verdict['qualified'] is False and verdict['provider_calls'] == 0
    assert verdict['inference_artifact_exported'] is False
    assert (BASE / 'before-pins.json').read_bytes() == (BASE / 'after-pins.json').read_bytes()
    controls = json.loads((BASE / 'controls-03.json').read_text())
    assert controls['returncode'] == 0
    assert 'Ran 30 tests' in (BASE / 'controls-03.log').read_text()
    for name, digest in controls['source_sha256'].items():
        assert sha(BASE / name) == digest, name
    analysis = json.loads((BASE / 'resource-analysis.json').read_text())
    assert analysis['events_sha256'] == sha(BASE / 'native-resource-events.jsonl')
    assert analysis['event_count'] == 18
    assert analysis['failed_replay_preceding_observation']['anon_delta_bytes'] == 76845056
    OUT.mkdir()
    copies = []
    for name in NAMES:
        path = BASE / name
        assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        assert path.stat().st_size <= 262144, name
        path.read_text()  # Public whitelist contains text only.
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copies.append(dict(path=name, artifact=str(path), bytes=path.stat().st_size, sha256=sha(path)))
    write('provenance.json', dict(schema='explicit-whitelist-provenance@1', copies=copies,
        excluded=['authentication', 'benchmark source bodies', 'hidden verifier', 'model weights',
                  'native inference bodies', 'databases', 'wheels', 'binaries', 'runtime archive',
                  'full deployment inventories', 'earlier control generations'],
        inference_artifact_exported=False, production_qualification_claimed=False))
    (OUT / 'README.md').write_text('''# Top128 wheel payload diagnostic

This instrumented run remains unqualified. Initial Source384 preparation returned, but the final initial-context reload failed in the second source observation around cold immutable replay. No inference artifact was exported, so model-load counts and decoded coverage remain unavailable. The initial preparation return is control-flow evidence; it is not an independently verified publication result. No provider or official verifier ran, and there is no new task or token score.

The exact preceding archive and corrected host deployer were retained. Advice completed for 26,905 archive files (3,914,232,094 logical bytes), four public Codex executable copies (628,736,528 bytes), and 131 installed payloads (565,702,269 bytes): three archive-pinned extension copies plus the 128 largest members selected from the pinned CPU wheel. No installed-tree discovery occurs. All selected bodies are independently pinned, and metadata fences, no-follow handling, bounds and hash-before-hint checks remain active. Logical bytes advised are not bytes reclaimed.

Observed available memory rose 6,764→8,229 MiB across archive advice, 8,229→8,846 MiB across executable advice, and 8,845→9,398 MiB across payload advice. The pre-context sample was 9,399 MiB. The same five-CPU/12-GiB profile, 6,144-MiB root reservation, 4,096-MiB child reservations and 2,458-MiB headroom remained active. No deadline or admission policy was changed.

The bounded source-free trace contains 18 stage events. During the first cold-replay observation, anonymous memory increased by 76,845,056 bytes (73.285 MiB), while file cache increased by only 4,096 bytes. The next observation entered with 8,561 MiB available and failed 30 seconds later with the same reported amount, below the 8,602-MiB threshold. These are entry/error samples, not internal per-admission samples. The scheduler's post-unwind snapshot retained `proof_memory_headroom`; the separate failure-handling sample of 8,723 MiB was taken after owner teardown. The trace does not distinguish DuckDB buffers, Python allocations and allocator retention.

Preparation took 10.675 seconds, initial context lasted 177.945 seconds before failure, and the native probe lasted 189.732 seconds. Initial indexing and numerical-worker stages returned earlier in the trace. These timings are not a successful Source384 deadline qualification. All leases were released and the container was removed.

Thirty final helper/instrumentation controls passed. Earlier control generations are not combined with this count. Instrumentation wraps stages in memory, exports no source text or local variables, preserves primary errors, propagates timeout signals, and restores hooks. Production qualification is explicitly false. The uninstrumented production path and full task remain pending.

The package contains only the explicit public whitelist recorded in `provenance.json`. No task bodies, authentication contents, weights, native inference bodies, databases, binaries or hidden verifier data are included. Original local paths in receipts identify the retained experiment; `manifest.json` binds every included member. Earlier diagnostic packages remain separate and unchanged.
''')
    rows = [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(OUT.rglob('*')) if p.is_file()]
    write('manifest.json', dict(schema='closed-wheel-payload-diagnostic-evidence@1', files=rows))
    result = dict(manifest_sha256=sha(OUT / 'manifest.json'), members=len(rows),
                  member_bytes=sum(r['bytes'] for r in rows), files=[r['path'] for r in rows]+['manifest.json'])
    (BASE / 'scoped-files.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__ == '__main__':
    main()
