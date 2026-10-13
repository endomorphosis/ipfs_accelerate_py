import json
from pathlib import Path
from benchmarks.agent_supervisor.container_coding.terminal_source_view_policy import require_source_view_archive
from benchmarks.agent_supervisor.container_coding.terminal_semantic_metadata_policy import require_semantic_metadata_archive
from benchmarks.agent_supervisor.container_coding.terminal_deployment import validate_source384_binding, validate_torch_cpu_wheel_binding
from benchmarks.agent_supervisor.container_coding.terminal_grok_deployment import validate_grok_binding
manifest=json.loads(Path('/home/barberb/lift_coding/artifacts/source-context-substitution-20261013/live-preparation/runtime-01/manifest.json').read_text())
require_source_view_archive(manifest,'identity@1','source384-5cpu-20gib-formalization@1')
require_semantic_metadata_archive(manifest,'common-bindings@1')
assert validate_source384_binding(manifest)
assert validate_torch_cpu_wheel_binding(manifest)
assert validate_grok_binding(manifest,required=True)
print('Current source-view, metadata, Source384, Torch and Grok bindings verified.')
