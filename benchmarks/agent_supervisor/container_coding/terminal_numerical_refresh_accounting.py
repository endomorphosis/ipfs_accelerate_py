"""Retain current-call learned successor work separately from retrieval work."""

COUNTERS = ("native_worker_calls", "model_loads", "head_inference_calls", "inferred_units",
            "embedding_production_calls", "embedded_texts")
SCHEMA = "terminal-learned-numerical-refresh-accounting@1"


def _counts(value):
    if (type(value) is not dict
            or value.get("schema") != "supervisor-learned-numerical-refresh-execution@1"
            or value.get("scope") != "current_successor_refresh_call"
            or value.get("counters_known") is not True
            or any(type(value.get(k)) is not bool for k in ("reused", "native_worker_executed", "inference_executed"))
            or any(type(value.get(k)) is not int or not 0 <= value[k] <= 128 for k in COUNTERS[1:])):
        return None
    if (any(value[k] > 1 for k in ("model_loads", "head_inference_calls", "embedding_production_calls"))
            or value["inference_executed"] is not bool(value["head_inference_calls"])
            or bool(value["inferred_units"]) is not value["inference_executed"]
            or value["model_loads"] != value["embedding_production_calls"]
            or (value["embedded_texts"] > 0 and value["embedding_production_calls"] == 0)
            or value["embedded_texts"] > value["inferred_units"]
            or (value["inference_executed"] and value["embedded_texts"] == 0)
            or ((value["reused"] or not value["native_worker_executed"])
                and (value["inference_executed"] or any(value[k] for k in COUNTERS[1:])))
            or (value["reused"] and value["native_worker_executed"])):
        return None
    return dict(native_worker_calls=int(value["native_worker_executed"]),
                **{key: value[key] for key in COUNTERS[1:]})


def summarize(results):
    """A failed/unreceipted attempt remains unknown, never an invented zero."""
    rows = results if type(results) is list else []
    valid = [counts for row in rows if type(row) is dict
        and (counts := _counts(row.get("learned_numerical_refresh_execution"))) is not None]
    complete = bool(rows) and len(valid) == len(rows)
    return dict(schema=SCHEMA, scope="learned_numerical_successor_refresh",
        all_refreshes_receipted=complete, observed_refreshes=len(valid),
        unobserved_refreshes=len(rows)-len(valid),
        totals={key: sum(row[key] for row in valid) if complete else None for key in COUNTERS},
        known_subtotals={key: sum(row[key] for row in valid) if valid else None for key in COUNTERS},
        model_load_scope="GTE_construction", embedding_call_scope="native_embedding_production_API",
        task_completion_is_inference_evidence=False)


def project(value):
    """Bound comparison output; do not turn subtotals into complete costs."""
    if (type(value) is not dict or value.get("schema") != SCHEMA
            or value.get("scope") != "learned_numerical_successor_refresh"):
        return None
    result = summarize(None)
    complete = value.get("all_refreshes_receipted") is True
    for group in ("totals", "known_subtotals"):
        mapping = value.get(group)
        mapping = mapping if type(mapping) is dict else {}
        result[group] = {key: mapping[key] if type(mapping.get(key)) is int and mapping[key] >= 0
            and (group != "totals" or complete) else None for key in COUNTERS}
    for key in ("observed_refreshes", "unobserved_refreshes"):
        result[key] = value[key] if type(value.get(key)) is int and value[key] >= 0 else None
    totals = result["totals"]
    result["all_refreshes_receipted"] = (complete and bool(result["observed_refreshes"])
        and result["unobserved_refreshes"] == 0 and all(v is not None for v in totals.values())
        and totals == result["known_subtotals"]
        and totals["head_inference_calls"] <= totals["model_loads"] <= totals["native_worker_calls"] <= result["observed_refreshes"]
        and totals["model_loads"] == totals["embedding_production_calls"]
        and totals["embedded_texts"] <= totals["inferred_units"] <= 128 * totals["head_inference_calls"])
    if not result["all_refreshes_receipted"]:
        result["totals"] = dict.fromkeys(COUNTERS)
    return result
