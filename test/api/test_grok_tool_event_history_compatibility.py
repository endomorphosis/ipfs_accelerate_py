"""Read only the checked-in closed diagnostic, never private provider inputs."""
import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.cli_runtime.grok_stream_capture import validate_grok_stream_progress
from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations


def test_retained_v4_diagnostic_keeps_its_original_v1_coverage_and_counts():
    path = Path(__file__).resolve().parents[2] / (
        "docs/agent_supervisor/evidence/tool-event-census-20261015/live/actual-public-diagnostic-01.json")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "45bac6a736500c756d46d950494ec63f8b7cb0c93feaf10905ba30cae98da6b9"
    progress = json.loads(raw)["observation"]["invocations"][0]["native_grok_progress"]
    validated = validate_grok_stream_progress(progress)
    assert validated == progress
    assert validated["schema"] == "native-grok-stream-progress@4"
    tools = validated["tool_event_observations"]
    assert tools["schema"] == "native-grok-tool-events@1"
    assert tools["observation_complete"] is False
    assert tools["payload_node_limit"] == 4096
    assert sum(row["output"]["state"] == "oversized" for row in tools["rows"]) == 7
    assert tools["payloads"]["output"]["canonical_json_bytes"] == 169546
    assert all("bound_reason" not in row["output"] and "resolved_tool" not in row for row in tools["rows"])
    metric, = _invocations({"provider_invocations": [{"invocation_id": "closed-historical-compatibility",
        "phase": "coding", "provider": "grok_cli", "native_grok_progress": progress}]})
    assert metric["native_grok_progress"] == progress
    assert all(value is None for value in metric["tokens"].values())
    # Compatibility validation does not change the retained historical evidence.
    assert path.read_bytes() == raw
