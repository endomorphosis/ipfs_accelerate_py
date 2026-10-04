"""Pure structural contract reference with one representative bank for both buckets.

This is a candidate repair for the observed independent-cap defect. It neither
executes a task nor claims performance thresholds or formal source equivalence.
"""
from __future__ import annotations

from collections import Counter


class BatchingContractError(ValueError):
    """An input or output is outside the explicitly closed structural contract."""


def _inputs(buckets):
    if type(buckets) is not list or len(buckets) != 2 or any(type(b) is not list for b in buckets):
        raise BatchingContractError("exactly two request lists required")
    if sum(map(len, buckets)) > 100000:
        raise BatchingContractError("request count exceeds closed contract bound")
    ids = set()
    for bucket in buckets:
        for row in bucket:
            if (type(row) is not dict or type(row.get("request_id")) is not str or
                    not row["request_id"] or len(row["request_id"]) > 1024 or
                    any(type(row.get(k)) is not int or not 0 <= row[k] <= 2**63-1 for k in ("prompt_len", "gen_len"))):
                raise BatchingContractError("unique identifiers and exact nonnegative 63-bit token integers required")
            if row["request_id"] in ids:
                raise BatchingContractError("duplicate request identifier across input buckets")
            ids.add(row["request_id"])


def build_global_shape_contract_reference(buckets):
    """Build a fresh plan using integer-only endpoints and globally shared shapes."""
    _inputs(buckets)
    unique = sorted({((r["prompt_len"] + 63) // 64) * 64 for b in buckets for r in b})
    count = min(8, len(unique))
    # Each endpoint is an observed aligned length; the final endpoint is the
    # maximum. Thus an eligible representative exists for every admitted input.
    reps = [unique[((i + 1) * len(unique) - 1) // count] for i in range(count)]
    output = []
    for index, bucket in enumerate(buckets, 1):
        records = []
        for request in bucket:
            aligned = ((request["prompt_len"] + 63) // 64) * 64
            representative = next(r for r in reps if r >= aligned)
            # Separate generation lengths avoid decode padding in this reference;
            # batch overhead/latency tradeoffs need the public cost model checks.
            records.append({"request_id": request["request_id"],
                "batch_id": f"bucket-{index}-s-{representative}-g-{request['gen_len']}",
                "shape": {"seq_align": representative, "heads_align": 32, "hidden_align": 4096}})
        output.append(records)
    validate_global_shape_contracts(buckets, output)
    return {"schema": "terminal-batching-global-shape-reference@1", "plans": output,
        "shared_representatives": reps, "structural_contracts_checked": True,
        "performance_thresholds_checked": False, "full_task_satisfaction": "unknown",
        "source_equivalence_proved": False, "training_calls": 0, "checker_calls": 0,
        "proof_authority": False, "execution_authority": False, "completion_authority": False}


def validate_global_shape_contracts(buckets, plans):
    """Check structural postconditions independently of representative selection."""
    _inputs(buckets)
    if type(plans) is not list or len(plans) != 2 or any(type(p) is not list for p in plans):
        raise BatchingContractError("exactly two plan lists required")
    shapes = set(); batch_shapes = {}
    for inputs, plan in zip(buckets, plans):
        if len(plan) != len(inputs):
            raise BatchingContractError("plan record count differs from input count")
        by_id = {r["request_id"]: r for r in inputs}
        counts = Counter()
        for row in plan:
            if (type(row) is not dict or type(row.get("request_id")) is not str or
                    row["request_id"] not in by_id or type(row.get("batch_id")) is not str or
                    not row["batch_id"] or type(row.get("shape")) is not dict):
                raise BatchingContractError("bounded source-associated output records required")
            shape = row["shape"]
            if any(type(shape.get(k)) is not int for k in ("seq_align", "heads_align", "hidden_align")):
                raise BatchingContractError("exact integer shape components required")
            triple = tuple(shape[k] for k in ("seq_align", "heads_align", "hidden_align"))
            request = by_id[row["request_id"]]
            if (triple[0] % 64 or triple[0] < ((request["prompt_len"] + 63) // 64) * 64 or
                    triple[1:] != (32, 4096)):
                raise BatchingContractError("shape violates request alignment/coverage contract")
            if row["batch_id"] in batch_shapes and batch_shapes[row["batch_id"]] != triple:
                raise BatchingContractError("batch identifier denotes inconsistent shapes")
            batch_shapes[row["batch_id"]] = triple; shapes.add(triple); counts[row["request_id"]] += 1
        if counts != Counter(by_id.keys()):
            raise BatchingContractError("requests are not included exactly once")
    if len(shapes) > 8:
        raise BatchingContractError("global unique shape count exceeds eight")
    return {"global_unique_shape_count": len(shapes), "input_request_count": sum(map(len, buckets)),
        "included_exactly_once": True, "shapes_aligned_and_cover_requests": True,
        "batch_identifiers_have_consistent_shapes": True, "scope": "structural_only_performance_unchecked"}
