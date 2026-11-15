"""Authored native receipts exercise the real Harbor accounting consumer.

No provider runs or native artifacts are authenticated by these fixtures.
"""
from copy import deepcopy
import hashlib

import pytest

from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import (
    AgentContext, measured_usage, populate_harbor_usage,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _codex_receipt(invocation="one", *, thread=None, inputs=100, outputs=10,
                   cached=80, records=1, snapshot=None, complete=True,
                   status="provider_returned"):
    """Complete authored shape; all IDs and hashes are fixture observations."""
    thread = thread or "authored-native-" + invocation
    return dict(schema="router-implementation-invocation@1", invocation_id=invocation,
        provider="codex_cli", model="gpt-6.1-sol", purpose="coding", status=status,
        workspace="/authored/workspace", native_rollout_usage=dict(
            schema="native-codex-cumulative-usage@1", thread_id=thread,
            rollout_sha256=_sha(snapshot or thread + ":" + str(records)),
            observed_token_count_records=records, usage_available=True,
            task_complete_observed=complete, cache_included_in_input=True,
            usage=dict(input_tokens=inputs, cached_input_tokens=cached,
                output_tokens=outputs, total_tokens=inputs + outputs),
            billing_total_verified=False))


def _grok_receipt(invocation="one", *, complete=True, writes=5):
    return dict(schema="router-implementation-invocation@1", invocation_id=invocation,
        provider="grok_cli", model="grok-4.7", purpose="coding", status="provider_returned",
        workspace="/authored/workspace", native_rollout_usage=dict(
            schema="native-grok-final-usage@1", usage_available=True,
            totals_scope="native_final_envelope", usage_complete_observed=complete,
            usage_observation_sha256=_sha("authored-envelope-" + invocation),
            observed_final_usage_records=1, task_complete_observed=True,
            cache_included_in_input=False,
            usage=dict(input_tokens=10, cached_input_tokens=30,
                cache_write_input_tokens=writes, output_tokens=2,
                total_tokens=42 + writes), billing_total_verified=False))


def test_harbor_uses_latest_resumed_session_and_retains_observed_invocation_count():
    first = _codex_receipt("first", thread="authored-session-123456789", complete=False)
    second = _codex_receipt("second", thread="authored-session-123456789", inputs=150,
        outputs=20, cached=100, records=2)
    usage = measured_usage({"provider_invocations": [first, second, deepcopy(second)]})
    context = AgentContext()
    populate_harbor_usage(context, usage)
    assert usage["provider_calls"] == 2
    assert usage["total_tokens"] == 170
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (150, 100, 20)
    assert usage["usage_accounting_complete"] is True
    assert usage["billing_total_verified"] is False


@pytest.mark.parametrize("change", ["counter_regression", "equal_record_conflict", "workspace"])
def test_harbor_refuses_conflicting_snapshots_of_one_native_session(change):
    first = _codex_receipt("first", thread="authored-session-123456789")
    second = _codex_receipt("second", thread="authored-session-123456789", inputs=150,
        outputs=20, cached=100, records=2)
    if change == "counter_regression":
        second["native_rollout_usage"]["usage"].update(input_tokens=90, cached_input_tokens=70,
            total_tokens=110)
    elif change == "equal_record_conflict":
        second["native_rollout_usage"]["observed_token_count_records"] = 1
    else:
        second["workspace"] = "/authored/different-workspace"
    usage = measured_usage({"provider_invocations": [first, second]})
    context = AgentContext(n_input_tokens=999, n_cache_tokens=99, n_output_tokens=9)
    populate_harbor_usage(context, usage)
    assert usage["provider_calls"] == 2
    assert usage["total_tokens"] is None
    assert usage["harbor_usage_exportable"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None


@pytest.mark.parametrize("field", ["thread_id", "rollout_sha256", "observed_token_count_records"])
def test_harbor_does_not_invent_native_session_custody(field):
    row = _codex_receipt()
    del row["native_rollout_usage"][field]
    usage = measured_usage({"provider_invocations": [row]})
    context = AgentContext()
    populate_harbor_usage(context, usage)
    assert usage["total_tokens"] is None
    assert usage["usage_accounting_complete"] is False
    assert context.n_input_tokens is None


def test_harbor_does_not_export_a_raw_input_category_for_mixed_provider_semantics():
    usage = measured_usage({"provider_invocations": [_codex_receipt(), _grok_receipt("grok")]})
    context = AgentContext()
    populate_harbor_usage(context, usage)
    assert usage["input_tokens"] is None
    assert usage["harbor_usage_exportable"] is False
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None


@pytest.mark.parametrize("change", ["cached_larger_than_input", "wrong_total", "bool_counter",
                                    "missing_counter", "wrong_scope"])
def test_harbor_rechecks_export_category_conservation(change):
    usage = measured_usage({"provider_invocations": [_codex_receipt()]})
    assert usage["harbor_usage_exportable"] is True
    if change == "cached_larger_than_input":
        usage["cached_input_tokens"] = 101
    elif change == "wrong_total":
        usage["total_tokens"] = 111
    elif change == "bool_counter":
        usage["output_tokens"] = True
    elif change == "missing_counter":
        del usage["cached_input_tokens"]
    else:
        usage["token_count_scope"] = "unknown"
    context = AgentContext(n_input_tokens=999, n_cache_tokens=99, n_output_tokens=9)
    populate_harbor_usage(context, usage)
    assert context.n_input_tokens is context.n_cache_tokens is context.n_output_tokens is None
