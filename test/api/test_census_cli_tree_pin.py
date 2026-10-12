"""Real formalization imports and census CLI preserve the strict JSON pipe.

Authored native capsules and Git worktrees are real. Child provider operations
are forbidden. Repeated AutoformalSession clients exercise the actual datasets
tree check before the actual census CLI; no Source384 inference is claimed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_router_integration import allocated  # noqa: F401
from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_semantic_instruction_dispatch import expanded_selected, compose  # noqa: F401
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _repair_json


@pytest.mark.parametrize("foreign_tree", [False, True])
def test_real_census_cli_stdout_remains_strict_json_after_logic_tree_imports(allocated, tmp_path, foreign_tree):
    from ipfs_datasets_py.logic.autoformal import tree_pin
    import ipfs_accelerate_py
    import ipfs_kit_py

    repository, workspace, prompt, _source = allocated
    roots = [str(Path(ipfs_accelerate_py.__file__).resolve().parent.parent),
             str(tree_pin.workspace_root()), str(Path(ipfs_kit_py.__file__).resolve().parent.parent)]
    foreign = tmp_path / "foreign_parser.py"
    foreign.write_text("# parser from an unrelated checkout\n")
    script = "import sys\nsys.path[:0]=" + repr(roots) + "\n" + '''
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import intelligence_index
def forbidden(*args, **kwargs):
    raise AssertionError("census attempted provider dispatch")
llm_router.generate_text = llm_router.get_llm_provider = forbidden
intelligence_index.discover_available_providers = intelligence_index.select_efficient_route = forbidden
from ipfs_datasets_py.logic.autoformal import AutoformalSession
'''
    if foreign_tree:
        script += "from ipfs_datasets_py.logic.deontic.utils import deontic_parser\n"
        script += "deontic_parser.__file__=" + repr(str(foreign)) + "\n"
    script += '''
for _ in range(3):
    AutoformalSession()
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main
raise SystemExit(main())
'''
    argv = [sys.executable, "-B", "-c", script, "--provider", "codex_cli", "--model", "authored-model",
        "--semantic-repository", str(repository), "--semantic-metadata-view", "common-bindings@1",
        "--semantic-metadata-census-only", "--coding-reply-mode", "ordinary-completion@1"]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI="0", PYTHONDONTWRITEBYTECODE="1")
    for key in ("OPENAI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(key, None)
    call = subprocess.run(argv, input=prompt.encode(), cwd=workspace, capture_output=True,
        env=environment, timeout=90)
    assert b"census attempted provider dispatch" not in call.stderr
    if foreign_tree:
        assert call.returncode != 0
        assert call.stdout == b""
        assert b"LogicTreePinError" in call.stderr
        return
    assert call.returncode == 0, call.stderr.decode()
    observed = _repair_json(call.stdout)
    assert observed == json.loads(call.stdout)
    assert len(call.stdout.splitlines()) == 1
    assert observed["schema"] == "router-semantic-metadata-census@1"
    assert observed["current_semantic_encoder_used"] is True
    assert observed["provider_calls"] == observed["router_calls"] == 0
    assert observed["provider_dispatch_observed"] is False
    # Importing the full AutoformalSession stack performs another tree check;
    # the signed zero-session case below separately observes the bare CLI path.
    assert len([line for line in call.stderr.splitlines() if line.startswith(b"logic tree pin ")]) == 4
    # The consumer continues rejecting contaminated or concatenated records;
    # moving diagnostics must never become permissive JSON prefix stripping.
    for malformed in (b"diagnostic\n" + call.stdout, call.stdout + call.stdout):
        with pytest.raises(ValueError):
            _repair_json(malformed)


@pytest.mark.parametrize("explicit_sessions", [0, 3])
def test_real_signed_instruction_census_cli_preserves_json(
        expanded_selected, monkeypatch, explicit_sessions, record_property):
    """Exercise signed admission and the actual CLI without any model dispatch.

    The zero-session case observes the unmodified CLI path for this authored
    native capsule. The repeated-session case exercises the diagnostic protocol
    independently; these fixtures do not claim a neural Source384 replay.
    """
    from ipfs_datasets_py.logic.autoformal import tree_pin
    import ipfs_accelerate_py
    import ipfs_kit_py

    case = compose(expanded_selected, monkeypatch)
    request = case["request"]
    roots = [str(Path(ipfs_accelerate_py.__file__).resolve().parent.parent),
             str(tree_pin.workspace_root()), str(Path(ipfs_kit_py.__file__).resolve().parent.parent)]
    script = "import sys\nsys.path[:0]=" + repr(roots) + "\n" + '''
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.llm_allocation import intelligence_index
def forbidden(*args, **kwargs):
    raise AssertionError("census attempted provider dispatch")
llm_router.generate_text = llm_router.get_llm_provider = forbidden
intelligence_index.discover_available_providers = intelligence_index.select_efficient_route = forbidden
'''
    if explicit_sessions:
        script += "from ipfs_datasets_py.logic.autoformal import AutoformalSession\n"
        script += f"for _ in range({explicit_sessions}):\n    AutoformalSession()\n"
    script += '''
import json
from pathlib import Path
traces = []
def trace_tree_pin(frame, event, arg):
    if event != "call" or frame.f_code.co_name != "require_workspace_logic_tree":
        return
    stack = []
    parent = frame.f_back
    while parent is not None:
        if "ipfs_" in parent.f_code.co_filename:
            stack.append([Path(parent.f_code.co_filename).name, parent.f_code.co_name])
        parent = parent.f_back
    traces.append(stack)
sys.setprofile(trace_tree_pin)
try:
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import main
    raise SystemExit(main())
finally:
    sys.setprofile(None)
    print("tree-pin natural call trace " + json.dumps(traces), file=sys.stderr)
'''
    argv = [sys.executable, "-B", "-c", script, "--provider", "codex_cli", "--model", "authored-model",
        "--semantic-repository", str(case["root"]),
        "--semantic-metadata-view", "common-bindings-instruction@1",
        "--semantic-metadata-census-only", "--coding-reply-mode", "ordinary-completion@1",
        "--public-instruction-artifact", str(request["artifact"]),
        "--public-instruction-sha256", request["expected_sha256"],
        "--public-instruction-task-cid", request["task_cid"]]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI="0", PYTHONDONTWRITEBYTECODE="1")
    for key in ("OPENAI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(key, None)
    call = subprocess.run(argv, input=request["prompt"].encode(), cwd=request["workspace"],
        capture_output=True, env=environment, timeout=90)
    assert call.returncode == 0, call.stderr.decode()
    assert b"census attempted provider dispatch" not in call.stderr
    observed = _repair_json(call.stdout)
    assert len(call.stdout.splitlines()) == 1
    assert observed["schema"] == "router-semantic-metadata-census@1"
    assert observed["current_semantic_encoder_used"] is True
    assert observed["provider_calls"] == observed["router_calls"] == 0
    assert observed["provider_dispatch_observed"] is False
    assert observed["public_instruction_included"] is True
    assert len([line for line in call.stderr.splitlines() if line.startswith(b"logic tree pin ")]) == (
        explicit_sessions + (1 if explicit_sessions else 0))
    prefix = b"tree-pin natural call trace "
    trace_rows = [line[len(prefix):] for line in call.stderr.splitlines() if line.startswith(prefix)]
    assert len(trace_rows) == 1
    traces = json.loads(trace_rows[0])
    assert traces == []
    record_property("natural_tree_pin_calls_for_authored_capsule", len(traces))
