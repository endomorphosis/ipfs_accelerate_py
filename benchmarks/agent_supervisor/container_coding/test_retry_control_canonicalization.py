"""Canonicalize only known Harbor sets before freezing new trial controls."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_controls import (
    canonicalize_serialized_retry_sets, build_controls, observe_controls,
    compare_controls,
)
from benchmarks.agent_supervisor.container_coding.native_codex_baseline import config_for


HASHES = {"instruction.md": "a" * 64}


def observe(config):
    declared = build_controls(config, task_input_sha256=HASHES,
        task="fix-code-vulnerability", model="gpt-6.1-sol",
        reasoning_effort="high", cli_version="0.160.0")
    return observe_controls({"comparison_controls": declared}, config,
        current_task_hashes=HASHES)


def test_only_known_top_level_retry_sets_are_ordered_without_mutating_input():
    original = {"retry": {"include_exceptions": ["B", "A"],
        "exclude_exceptions": ["D", "C"], "ordered_future_field": ["z", "a"]},
        "extra_instructions": ["second", "first"],
        "agents": [{"kwargs": {"argv": ["z", "a"],
            "retry": {"exclude_exceptions": ["nested-z", "nested-a"]}}}]}
    saved = deepcopy(original)
    actual = canonicalize_serialized_retry_sets(original)
    assert original == saved
    assert actual["retry"]["include_exceptions"] == ["A", "B"]
    assert actual["retry"]["exclude_exceptions"] == ["C", "D"]
    actual["retry"]["include_exceptions"] = saved["retry"]["include_exceptions"]
    actual["retry"]["exclude_exceptions"] = saved["retry"]["exclude_exceptions"]
    assert actual == original


@pytest.mark.parametrize("config", [{}, {"retry": None}, {"retry": {}},
    {"retry": {"include_exceptions": None, "exclude_exceptions": []}}])
def test_missing_null_and_empty_filters_keep_their_meaning(config):
    assert canonicalize_serialized_retry_sets(config) == config


@pytest.mark.parametrize("value", ["Error", [1], ["Error", "Error"]])
def test_malformed_serialized_filters_are_not_silently_repaired(value):
    with pytest.raises(ValueError, match="unique string lists"):
        canonicalize_serialized_retry_sets({"retry": {"exclude_exceptions": value}})


def test_unsorted_historical_v1_declaration_still_replays_exactly():
    config = config_for(Path("/dataset"), Path("/out"))
    config["retry"]["exclude_exceptions"] = ["B", "A"]
    observed = observe(config)
    assert observed["declared"]["schema"] == "terminal-benchmark-declared-controls@1"
    assert observed["configuration_unchanged"] is True
    changed = canonicalize_serialized_retry_sets(config)
    assert observe_controls({"comparison_controls": observed["declared"]},
        changed, current_task_hashes=HASHES)["status"] == "mismatch"


def test_instruction_order_remains_a_visible_control_difference():
    config = config_for(Path("/dataset"), Path("/out"))
    config["extra_instructions"] = ["first", "second"]
    changed = deepcopy(config)
    changed["extra_instructions"].reverse()
    result = compare_controls(observe(canonicalize_serialized_retry_sets(config)),
        observe(canonicalize_serialized_retry_sets(changed)))
    assert result["matches"] is False
    assert result["differences"] == ["job_controls_sha256.extra_instructions"]


def test_separate_hash_seed_processes_produce_identical_retry_controls():
    pytest.importorskip("harbor.models.job.config")
    root = Path(__file__).resolve().parents[3]
    script = """
import json
from pathlib import Path
from harbor.models.job.config import JobConfig
from benchmarks.agent_supervisor.container_coding.benchmark_controls import canonicalize_serialized_retry_sets, build_controls
from benchmarks.agent_supervisor.container_coding.native_codex_baseline import config_for
config = config_for(Path('/dataset'), Path('/out'))
wire = canonicalize_serialized_retry_sets(JobConfig.model_validate(config, extra='forbid').model_dump(mode='json'))
reloaded = JobConfig.model_validate(wire, extra='forbid')
assert set(wire['retry']['exclude_exceptions']) == reloaded.retry.exclude_exceptions
assert wire['retry']['include_exceptions'] is None
controls = build_controls(wire, task_input_sha256={'instruction.md': 'a'*64}, task='fix-code-vulnerability', model='gpt-6.1-sol', reasoning_effort='high', cli_version='0.160.0')
print(json.dumps({'wire': wire, 'controls': controls}, sort_keys=True))
"""
    rows = []
    for seed in ("1", "2"):
        process = subprocess.run([sys.executable, "-B", "-c", script],
            cwd=root, env={**os.environ, "PYTHONHASHSEED": seed,
                "PYTHONDONTWRITEBYTECODE": "1"}, text=True,
            capture_output=True, check=True, timeout=30)
        rows.append(json.loads(process.stdout.splitlines()[-1]))
    assert rows[0] == rows[1]
