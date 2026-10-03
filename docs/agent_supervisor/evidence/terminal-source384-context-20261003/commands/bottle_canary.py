"""Retained public-source canary, not a new Terminal Bench trial or score."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding

base = Path(os.environ.get('SOURCE384_CANARY_OUTPUT', '/home/barberb/lift_coding/artifacts/terminal-source384-context-20261003/bottle-canary-03'))
base.mkdir()
root = base / 'app'
root.mkdir()
retained = Path('/home/barberb/lift_coding/artifacts/terminal_bench_supervisor/full-integration-20260929/terminal-full-preflight-host-01/app/bottle.py')
original_receipt = retained.parent.parent / 'state/original-image.json'
expected = '761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba'
raw = retained.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
assert json.loads(original_receipt.read_bytes())['sources']['bottle.py']['sha256'] == expected
(root / 'bottle.py').write_bytes(raw)
for args in (('init', '-q'), ('add', 'bottle.py'), ('-c', 'user.name=Source canary', '-c', 'user.email=canary@example.invalid', 'commit', '-qm', 'Retained original public Bottle bytes')):
    subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
instruction = base / 'instruction.md'
shutil.copyfile('/home/barberb/lift_coding/.benchmarks/terminal-bench-2/fix-code-vulnerability/instruction.md', instruction)
checkpoint = Path(os.environ['CODEBASE384_CHECKPOINT'])
snapshot = os.environ['CODEBASE384_EMBEDDING_SNAPSHOT']
config = dict(schema='terminal-source384-config@1', mode='pinned_parent',
    checkpoint_path=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    embedding_snapshot=snapshot, embedding_revision=embedding.PINNED_REVISION,
    embedding_assets=embedding._snapshot_assets(snapshot)[1], training_steps=0, download_calls=0)
config_path = base / 'source384-config.json'
config_path.write_text(json.dumps(config, sort_keys=True))
report = dict(schema='retained-bottle-source384-canary@1', source_sha256=expected,
    source_path=str(retained), original_image_receipt=str(original_receipt),
    original_image_receipt_sha256=hashlib.sha256(original_receipt.read_bytes()).hexdigest(),
    fresh_docker_input_qualified=False, official_reward=None, benchmark_score=False,
    text_provider_calls=0, training_steps=0)
started = time.monotonic()
try:
    prepared = prep.prepare(repository=root, instruction=instruction, state=base / 'state')
    report['initial_context'] = prep.initial_context(state=base / 'state', source384_config=config_path)
    report['status'] = 'prepared'
except Exception as exc:
    report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
    raise
finally:
    report['seconds'] = time.monotonic()-started
    (base / 'canary.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','seconds','source_sha256','benchmark_score')}))
