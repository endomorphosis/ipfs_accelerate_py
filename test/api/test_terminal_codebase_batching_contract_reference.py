"""Global-cap repair and independent structural postcondition controls."""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_codebase_batching_contract_reference as api


def inputs(count):
    return [[{"request_id": f"bucket-{b}-r-{i}", "prompt_len": (i + b * count) * 64 + 1,
        "gen_len": i % 5} for i in range(count)] for b in range(2)]


@pytest.mark.parametrize("count", [0, 1, 4, 8, 9, 16, 43])
def test_joint_representatives_satisfy_global_cap_and_preserve_inputs(count):
    buckets = inputs(count); before = deepcopy(buckets)
    result = api.build_global_shape_contract_reference(buckets)
    checked = api.validate_global_shape_contracts(buckets, result["plans"])
    assert checked["global_unique_shape_count"] <= 8
    assert checked["input_request_count"] == 2 * count
    assert buckets == before
    assert len(result["shared_representatives"]) == min(8, 2 * count)
    assert result["full_task_satisfaction"] == "unknown"
    assert result["performance_thresholds_checked"] is False


def test_two_disjoint_eight_shape_buckets_use_one_global_bank():
    buckets = [[{"request_id": f"b{b}-r{i}", "prompt_len": (b * 8 + i + 1) * 64,
        "gen_len": 1} for i in range(8)] for b in range(2)]
    result = api.build_global_shape_contract_reference(buckets)
    assert result["shared_representatives"] == list(range(128, 1025, 128))
    assert api.validate_global_shape_contracts(buckets, result["plans"])["global_unique_shape_count"] == 8


@pytest.mark.parametrize("change", ["missing", "duplicate", "global-cap", "unaligned", "under-cover", "heads", "batch-collision"])
def test_foreign_plan_postconditions_refused(change):
    buckets = inputs(8); plans = api.build_global_shape_contract_reference(buckets)["plans"]
    if change == "missing": plans[0].pop()
    elif change == "duplicate": plans[0][1] = deepcopy(plans[0][0])
    elif change == "global-cap":
        for b, plan in enumerate(plans):
            for i, row in enumerate(plan): row["shape"]["seq_align"] = (100 + b * 8 + i) * 64
    elif change == "unaligned": plans[0][0]["shape"]["seq_align"] += 1
    elif change == "under-cover": plans[0][0]["shape"]["seq_align"] = 0
    elif change == "heads": plans[0][0]["shape"]["heads_align"] = 31
    else:
        plans[1][-1]["batch_id"] = plans[0][0]["batch_id"]
    with pytest.raises(api.BatchingContractError): api.validate_global_shape_contracts(buckets, plans)


@pytest.mark.parametrize("change", ["duplicate-id", "bool-length", "negative-length"])
def test_invalid_input_domain_refused(change):
    buckets = inputs(2)
    if change == "duplicate-id": buckets[1][0]["request_id"] = buckets[0][0]["request_id"]
    elif change == "bool-length": buckets[0][0]["prompt_len"] = True
    else: buckets[0][0]["gen_len"] = -1
    with pytest.raises(api.BatchingContractError): api.build_global_shape_contract_reference(buckets)
