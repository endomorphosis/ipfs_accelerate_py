"""Genuine generic rendering, authored histories and offline local counters only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor import source_view_context_probe as subject
from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
from ipfs_accelerate_py.agent_supervisor.runtime.code_source_view import render_source_bytes
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter, render_model_prompt,
)

SOURCE = (b"# ORIGINAL_COMMENT_MUST_STAY_AUTHORITATIVE\nimport math\n\n"
    b"class Calculator:\n    def total(self, quantities: list[int]) -> int:\n"
    b"        return sum(quantities) + math.floor(2.0)\n")
BINDING = dict(task_id="offline-fixture", provider="grok_cli", model="fixture-model",
    tree_id="captured:fixture", acceptance_binding="unverified:fixture")
WORKSPACE = Path("/offline/authored-workspace")
PREFIX, SUFFIX = "PRESERVE_AUTHORED_REQUIREMENT", "\nUNVERIFIED_OFFLINE_SUFFIX"


def counter(callback=None, **changes):
    return PromptTokenCounter(provider="grok_cli", model="fixture-model", tokenizer_id="fixture-byte-counter",
        tokenizer_revision="fixture@1", special_token_policy="none",
        count_tokens=callback or (lambda text: len(text.encode())), **changes)


def probe(**changes):
    options = dict(source=SOURCE, path="src/real_code.py",
        binding=BINDING, common_prefix=PREFIX, common_suffix=SUFFIX, workspace=WORKSPACE,
        token_counter=counter(), rounds=3)
    options.update(changes)
    return subject.compare_source_view_context(**options)


def response(format):
    return subject._json(render_source_bytes("src/real_code.py", SOURCE, format=format)).decode() + "\n"


def rendered(history):
    return render_model_prompt(prompt=PREFIX + "\n" + subject.GUIDANCE + "\n".join(history) + SUFFIX,
        purpose="coding", workspace=WORKSPACE)[0]


def unknown(result):
    for comparison in result["comparisons"].values():
        assert comparison["comparison"]["cumulative_baseline_minus_candidate"] is None
        for arm in ("baseline", "candidate"):
            assert comparison[arm]["input_tokens"] is None
            assert comparison[arm]["estimated_input_tokens"] is None
            assert all(row["input_tokens"] is row["estimated_input_tokens"] is None
                for row in comparison[arm]["inputs"])


def test_real_generic_renderer_and_complete_metadata_and_instructions_are_counted():
    seen = []
    result = probe(token_counter=counter(lambda text: seen.append(text) or len(text.encode())))
    raw, compact = response("raw"), response("compact")
    assert b"import math" in SOURCE  # This is outside the separate arithmetic-only codec.
    assert all(text.startswith("Repository workspace:") or PREFIX in text for text in seen)
    repeated = result["comparisons"]["repeated_reads"]
    for arm, body in (("baseline", raw), ("candidate", compact)):
        expected = [len(rendered([body] * count).encode()) for count in (1, 2, 3)]
        assert [row["input_tokens"] for row in repeated[arm]["inputs"]] == expected
        assert repeated[arm]["input_tokens"] == sum(expected)
        assert all("source_sha256" in text and PREFIX in text and SUFFIX in text for text in seen)
    assert result["generic_view_has_alias_dictionary"] is False


def test_extra_expansion_call_charges_prior_compact_and_exact_response_replay():
    result = probe(rounds=2)
    expanded = result["comparisons"]["exact_source_expansion"]
    raw, compact = response("raw"), response("compact")
    assert expanded["baseline"]["input_count"] == 2
    assert expanded["candidate"]["input_count"] == 3
    assert [row["kind"] for row in expanded["candidate"]["inputs"]] == ["coding", "expansion", "coding"]
    assert expanded["baseline"]["input_tokens"] == 2 * len(rendered([raw]).encode())
    assert expanded["candidate"]["input_tokens"] == (
        len(rendered([compact]).encode()) + 2 * len(rendered([compact, raw]).encode()))


def test_actual_target_module_bytes_are_supported_without_importing_target():
    source = Path(subject.__file__).read_bytes()
    result = probe(source=source, path="benchmarks/agent_supervisor/source_view_context_probe.py", rounds=1)
    assert result["measurement_status"] == "measured"
    assert result["source_bytes"] == len(source)


def test_estimate_never_fills_exact_count_fields():
    result = probe(token_counter=counter(count_kind="estimate", compatibility_basis="authored-best-guess"))
    assert result["measurement_status"] == "estimated"
    for comparison in result["comparisons"].values():
        assert comparison["baseline"]["input_tokens"] is None
        assert comparison["baseline"]["estimated_input_tokens"] > 0
        assert comparison["tokenizer_compatibility_basis"] == "authored-best-guess"


def test_missing_counter_has_no_partial_or_inferred_totals():
    result = probe(token_counter=None)
    assert result["measurement_status"] == "unavailable"
    assert result["counter_evaluation_count"] == 0
    unknown(result)


@pytest.mark.parametrize("mode", ["exception", "provenance", "changing", "overflow"])
def test_later_comparison_failure_scrubs_both_comparisons(mode):
    token_counter, calls = counter(), {}

    def count(text):
        both = '"requested_format":"raw"' in text and '"requested_format":"compact"' in text
        calls[text] = calls.get(text, 0) + 1
        if both:
            if mode == "exception":
                raise RuntimeError("PRIVATE_EXCEPTION_AND_SOURCE_BODY")
            if mode == "provenance":
                object.__setattr__(token_counter, "compatibility_basis", "changed")
            if mode == "overflow":
                return 1 << 63
            return len(text.encode()) + calls[text]
        return len(text.encode())

    object.__setattr__(token_counter, "count_tokens", count)
    result = probe(token_counter=token_counter)
    assert result["measurement_status"] == "failed"
    assert result["all_comparison_counts_available"] is False
    assert "PRIVATE_EXCEPTION_AND_SOURCE_BODY" not in json.dumps(result)
    unknown(result)


def test_cancellation_preserves_original_instance():
    error = KeyboardInterrupt()
    def cancelled(text):
        raise error
    with pytest.raises(KeyboardInterrupt) as caught:
        probe(token_counter=counter(cancelled))
    assert caught.value is error


@pytest.mark.parametrize("changes", [dict(rounds=True), dict(rounds=0), dict(rounds=9),
    dict(path="../private.py"), dict(path="/private.py"), dict(path=".private/file.py"),
    dict(source="not bytes"), dict(workspace=Path("relative"))])
def test_invalid_contract_refuses_before_counter(changes):
    calls = []
    with pytest.raises(ValueError):
        probe(token_counter=counter(lambda text: calls.append(text) or 1), **changes)
    assert calls == []


def test_unsupported_source_and_excess_history_have_closed_rows_before_counting():
    calls = []
    for source, rounds, reason in ((b"def broken(:\n", 1, "source_view_unavailable"),
            (b"# " + b"x" * 60000 + b"\nx = 1\n", 8, "complete_input_bound")):
        result = probe(source=source, rounds=rounds, token_counter=counter(lambda text: calls.append(text) or 1))
        assert result["measurement_status"] == "unsupported" and result["failure_reason"] == reason
        assert result["comparisons"] is None
    assert calls == []


def test_input_bindings_remain_unchanged_and_output_has_no_bodies_or_authority():
    before = deepcopy(BINDING)
    result = probe()
    assert BINDING == before
    encoded = json.dumps(result)
    assert PREFIX not in encoded and SUFFIX not in encoded and SOURCE.decode() not in encoded
    assert "ORIGINAL_COMMENT_MUST_STAY_AUTHORITATIVE" not in encoded
    for flag in ("original_source_obligations_discharged", "actual_MCP_delivery", "native_history",
            "task_acceptance", "provider_invoked", "token_budget_guaranteed", "execution_authority", "completion_authority"):
        assert result[flag] is False
    assert result["whole_session_token_savings"] is None


def test_cli_captures_explicit_real_source_and_writes_fresh_private_report(tmp_path, monkeypatch, capsys):
    (tmp_path / "target.py").write_bytes(SOURCE)
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", lambda **kw: (
        counter(count_kind="estimate", compatibility_basis="offline-fixture"), {"offline": True, "count_kind": "estimate"}))
    output = tmp_path / "report.json"
    args = ["--root", str(tmp_path), "--paths", "target.py", "--output", str(output), "--model", "fixture-model"]
    assert subject.main(args) == 0
    saved = output.read_bytes()
    report = json.loads(saved)
    assert report["cases"][0]["measurement_status"] == "estimated"
    comparison = report["cases"][0]["comparisons"]["repeated_reads"]
    assert comparison["binding"]["tree_id"] == "sha256:" + report["population_captured_source_sha256"]
    assert report["population_tokens"] is None
    assert SOURCE not in saved
    assert output.stat().st_mode & 0o777 == 0o600
    assert subject.main(args) == 2 and output.read_bytes() == saved
    assert "ORIGINAL_COMMENT" not in capsys.readouterr().out


def test_cli_refuses_symlinks_and_private_errors_without_export(tmp_path, capsys):
    (tmp_path / "target.py").write_bytes(SOURCE)
    (tmp_path / "link.py").symlink_to("target.py")
    assert subject.main(["--root", str(tmp_path), "--paths", "link.py", "--model", "fixture-model",
        "--output", str(tmp_path / "new.json")]) == 2
    assert not (tmp_path / "new.json").exists()
    assert json.loads(capsys.readouterr().out)["reason"] == "invalid_input_or_local_failure"
