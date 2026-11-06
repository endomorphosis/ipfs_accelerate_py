"""Final native prompt-ledger accounting, using synthetic data and transport.

No native CLI, provider, credential, production catalog or billing endpoint is
used. The runner integration calls the actual runner with mocked routing and
transport; ledger observations grant no task or completion authority.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import ctypes.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test.api.test_grok_planning_temperature_zero import _fixture_library_resolver


class GrokPromptAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ipfs_accelerate_py.cli_runtime import cli_metadata, grok_native_usage, grok_prompt_accounting
        from ipfs_accelerate_py import llm_router
        from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner
        cls.accounting = grok_prompt_accounting
        cls.native_usage = grok_native_usage
        cls.metadata = cli_metadata
        cls.router = llm_router
        cls.runner = router_implementation_runner

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.stack.enter_context(patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(self.root),
            "IPFS_ACCEL_SKIP_CORE": "1", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
            "IPFS_AUTO_INSTALL": "false", "IPFS_DATASETS_AUTO_INSTALL": "false",
            "IPFS_ACCELERATE_LLM_ALLOCATION_DB": str(self.root / "allocation.duckdb"),
            "IPFS_ACCELERATE_META_INDEX_DUCKDB": str(self.root / "meta-index.duckdb"),
        }, clear=True))
        self.stack.enter_context(patch.object(subprocess, "Popen", side_effect=AssertionError("process forbidden")))
        self.stack.enter_context(patch.object(subprocess, "run", side_effect=AssertionError("process forbidden")))
        self.metadata.set_last_cli_observation("grok_cli", {})
        self.addCleanup(self.metadata.set_last_cli_observation, "grok_cli", {})

    @staticmethod
    def ledger():
        return {
            "type": "result", "text": "synthetic final answer", "stopReason": "end_turn",
            "num_turns": 3, "usage_is_incomplete": False, "cost_is_partial": False,
            "total_cost_usd_ticks": (1 << 53) + 17,
            "modelUsage": {
                "synthetic-primary-model": {
                    "inputTokens": 100, "cacheReadInputTokens": 20,
                    "cacheCreationInputTokens": 5, "outputTokens": 30,
                    "reasoningTokens": 7, "totalTokens": 130, "modelCalls": 2,
                },
                "synthetic-second-model": {"input_tokens": 9, "output_tokens": 4, "model_calls": 1},
            },
        }

    def project(self, payload):
        return self.accounting.project_grok_prompt_accounting(payload)

    @staticmethod
    def redigest(value):
        body = {key: item for key, item in value.items() if key != "projection_sha256"}
        value["projection_sha256"] = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False).encode()).hexdigest()
        return value

    def test_final_ledger_preserves_rounds_model_calls_categories_and_exact_ticks(self):
        result = self.project(self.ledger())
        self.assertEqual(result["schema"], "native-grok-prompt-accounting@1")
        self.assertEqual(result["scope"], "native_final_prompt_ledger")
        self.assertEqual(result["num_turns"], {"status": "valid", "value": 3})
        self.assertEqual(result["cost"], {
            "status": "valid", "observed_usd_ticks": (1 << 53) + 17,
            "usable_usd_ticks": (1 << 53) + 17, "ticks_per_usd": 10_000_000_000,
        })
        models = result["model_usage"]
        self.assertEqual((models["observed_rows"], models["retained_rows"], models["omitted_rows"]), (2, 2, 0))
        first = models["rows"][0]
        self.assertEqual(first["model_id_sha256"], hashlib.sha256(b"synthetic-primary-model").hexdigest())
        self.assertEqual(first["ordinal"], 1)
        self.assertEqual(first["counters"], {name: {"status": "valid", "value": value} for name, value in {
            "input_tokens": 100, "cached_input_tokens": 20, "cache_write_input_tokens": 5,
            "output_tokens": 30, "reasoning_output_tokens": 7, "total_tokens": 130,
            "model_calls": 2,
        }.items()})
        self.assertFalse(result["summation_permitted"])
        self.assertEqual(self.accounting.validate_grok_prompt_accounting(result), result)

    def test_partial_incomplete_and_missing_flags_withhold_usable_cost(self):
        for field in ("usage_is_incomplete", "cost_is_partial"):
            for kind in ("true", "missing", "invalid"):
                with self.subTest(field=field, kind=kind):
                    payload = self.ledger()
                    if kind == "missing":
                        del payload[field]
                    else:
                        payload[field] = True if kind == "true" else "false"
                    result = self.project(payload)
                    self.assertEqual(result[field]["status"], "valid" if kind == "true" else kind)
                    self.assertEqual(result["cost"]["observed_usd_ticks"], payload["total_cost_usd_ticks"])
                    self.assertIsNone(result["cost"]["usable_usd_ticks"])
                    self.assertFalse(result["billing_total_verified"])

    def test_absent_invalid_and_empty_payloads_keep_evidence_unknown(self):
        for payload, container in ((None, "missing"), ([], "invalid"), ("untrusted text", "invalid"), ({}, "mapping")):
            with self.subTest(container=container, payload_type=type(payload).__name__):
                result = self.project(payload)
                self.assertEqual(result["payload_container"], container)
                self.assertEqual(result["num_turns"], {"status": "missing", "value": None})
                self.assertEqual(result["usage_is_incomplete"], {"status": "missing", "value": None})
                self.assertEqual(result["cost_is_partial"], {"status": "missing", "value": None})
                self.assertIsNone(result["cost"]["usable_usd_ticks"])
                self.assertEqual(result["model_usage"]["rows"], [])
                self.assertIsNone(result["model_usage"]["observed_rows"])
                self.assertEqual(self.accounting.validate_grok_prompt_accounting(result), result)

    def test_invalid_scalars_never_become_counts_flags_or_tick_values(self):
        for field in ("num_turns", "total_cost_usd_ticks"):
            for value in (True, False, -1, 1.5, "2", None, 1 << 63):
                with self.subTest(field=field, value=repr(value)):
                    result = self.project({field: value})
                    scalar = result["cost"] if field == "total_cost_usd_ticks" else result[field]
                    self.assertEqual(scalar["status"], "invalid")
                    self.assertIsNone(scalar["observed_usd_ticks"] if field == "total_cost_usd_ticks" else scalar["value"])
        for field in ("usage_is_incomplete", "cost_is_partial"):
            for value in (0, 1, None, "false"):
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.project({field: value})[field], {"status": "invalid", "value": None})

    def test_model_counter_types_and_overflow_are_rejected_individually(self):
        for value in (True, False, -1, 0.5, "10", None, 1 << 63):
            with self.subTest(value=repr(value)):
                result = self.project({"modelUsage": {"unit-model": {"inputTokens": value, "modelCalls": value}}})
                counters = result["model_usage"]["rows"][0]["counters"]
                self.assertEqual(counters["input_tokens"], {"status": "invalid", "value": None})
                self.assertEqual(counters["model_calls"], {"status": "invalid", "value": None})
                self.assertEqual(result["model_usage"]["invalid_counter_fields"], 2)
        maximum = (1 << 63) - 1
        result = self.project({"modelUsage": {"unit-model": {"inputTokens": maximum, "modelCalls": 0}}})
        self.assertEqual(result["model_usage"]["rows"][0]["counters"]["input_tokens"]["value"], maximum)

    def test_matching_aliases_are_accepted_and_conflicts_are_not_chosen(self):
        row = {"inputTokens": 10, "input_tokens": 10, "prompt_tokens": 10, "modelCalls": 2, "model_calls": 2}
        result = self.project({"modelUsage": {"unit-model": row}})
        self.assertEqual(result["model_usage"]["rows"][0]["counters"]["input_tokens"], {"status": "valid", "value": 10})
        row["input_tokens"] = 11
        row["model_calls"] = 3
        result = self.project({"modelUsage": {"unit-model": row}})
        counters = result["model_usage"]["rows"][0]["counters"]
        self.assertEqual(counters["input_tokens"], {"status": "conflict", "value": None})
        self.assertEqual(counters["model_calls"], {"status": "conflict", "value": None})
        self.assertEqual(result["model_usage"]["conflicting_counter_fields"], 2)

    def test_float_dollars_are_not_converted_into_native_ticks(self):
        result = self.project({"total_cost_usd": 0.123456789, "usage_is_incomplete": False,
            "cost_is_partial": False, "modelUsage": {"unit-model": {"costUSD": 0.02}}})
        self.assertEqual(result["cost"]["status"], "missing")
        self.assertIsNone(result["cost"]["observed_usd_ticks"])
        self.assertIsNone(result["cost"]["usable_usd_ticks"])
        self.assertNotIn("dollar_cost", result)
        self.assertFalse(result["billing_total_verified"])

    def test_model_ids_are_hashed_and_arbitrary_provider_strings_are_not_copied(self):
        model = "PRIVATE_MODEL_IDENTIFIER_DO_NOT_COPY"
        marker = "RAW_PROVIDER_TEXT_DO_NOT_COPY"
        payload = {"text": marker, "message": marker, "prompt": marker,
                   "modelUsage": {model: {"inputTokens": 4, "description": marker}}}
        result = self.project(payload)
        encoded = json.dumps(result)
        self.assertNotIn(model, encoded)
        self.assertNotIn(marker, encoded)
        self.assertEqual(result["model_usage"]["rows"][0]["model_id_sha256"], hashlib.sha256(model.encode()).hexdigest())
        self.assertFalse(result["raw_provider_data_exported"])

    def test_invalid_and_oversized_model_ids_are_not_retained_or_stringified(self):
        class NeverStringify:
            def __str__(self):
                raise AssertionError("untrusted identity was stringified")
            __repr__ = __str__

        for identity in ("", "x" * 257, "model\nidentity", "model\x7fidentity", True, NeverStringify()):
            result = self.project({"modelUsage": {identity: {"modelCalls": 1}}})
            row = result["model_usage"]["rows"][0]
            self.assertEqual(row["identity_status"], "invalid")
            self.assertIsNone(row["model_id_sha256"])
            self.assertEqual(result["model_usage"]["invalid_identity_rows"], 1)

    def test_bounded_model_census_records_omissions_without_inspecting_them(self):
        class Omitted:
            def __str__(self):
                raise AssertionError("omitted row was inspected")
            __repr__ = __str__

        rows = {"unit-model-" + str(i): {"modelCalls": i} for i in range(32)}
        rows.update({Omitted(): Omitted() for _ in range(8)})
        result = self.project({"modelUsage": rows})
        models = result["model_usage"]
        self.assertEqual((models["observed_rows"], models["retained_rows"], models["omitted_rows"]), (40, 32, 8))
        self.assertTrue(models["truncated"])
        self.assertEqual([row["ordinal"] for row in models["rows"]], list(range(1, 33)))
        self.assertEqual(models["invalid_identity_rows"], 0)
        self.assertEqual(models["malformed_rows"], 0)
        self.assertLess(len(json.dumps(result).encode()), 65_536)

    def test_malformed_model_containers_and_rows_have_explicit_census(self):
        for native in (None, [], "untrusted model rows"):
            result = self.project({"modelUsage": native})
            self.assertEqual(result["model_usage"]["container"], "invalid")
            self.assertIsNone(result["model_usage"]["observed_rows"])
            self.assertEqual(result["model_usage"]["rows"], [])
        models = self.project({"modelUsage": {"unit-a": None, "unit-b": "opaque row"}})["model_usage"]
        self.assertEqual(models["malformed_rows"], 2)
        self.assertEqual(models["missing_counter_fields"], 14)
        self.assertTrue(all(counter == {"status": "missing", "value": None}
            for row in models["rows"] for counter in row["counters"].values()))

    def test_nested_tool_or_model_text_cannot_supply_final_ledger_fields(self):
        nested = self.ledger()
        result = self.project({"text": json.dumps(nested), "tool_result": nested,
            "usage": nested, "messages": [{"role": "assistant", "content": nested}],
            "modelCalls": {"unit-model": 99}})
        self.assertEqual(result["num_turns"]["status"], "missing")
        self.assertEqual(result["model_usage"]["container"], "missing")
        self.assertEqual(result["cost"]["status"], "missing")
        row = self.project({"modelUsage": {"unit-model": {"text": {"input_tokens": 999}}}})["model_usage"]["rows"][0]
        self.assertEqual(row["counters"]["input_tokens"]["status"], "missing")

    def test_projection_digest_detects_mutation_and_validation_returns_a_copy(self):
        original = self.project(self.ledger())
        digest = original["projection_sha256"]
        self.assertEqual(self.redigest(deepcopy(original))["projection_sha256"], digest)
        validated = self.accounting.validate_grok_prompt_accounting(original)
        self.assertEqual(validated, original)
        self.assertIsNot(validated, original)
        self.assertIsNot(validated["model_usage"]["rows"], original["model_usage"]["rows"])
        validated["num_turns"]["value"] = 12
        self.assertEqual(original["num_turns"]["value"], 3)
        with self.assertRaises(ValueError):
            self.accounting.validate_grok_prompt_accounting(validated)

    def test_rehashed_forgery_cannot_change_authority_typed_counts_or_cost_rules(self):
        payload = self.ledger()
        payload["num_turns"] = 1
        payload["modelUsage"] = {"unit-model": payload["modelUsage"]["synthetic-primary-model"]}
        original = self.project(payload)
        changes = (
            lambda value: value.update(billing_total_verified=True),
            lambda value: value["num_turns"].update(value=True),
            lambda value: value["model_usage"].update(retained_rows=True),
            lambda value: value["cost"].update(observed_usd_ticks=1.5),
            lambda value: value["cost_is_partial"].update(value=True),
        )
        for change in changes:
            candidate = deepcopy(original)
            change(candidate)
            self.redigest(candidate)
            with self.assertRaises(ValueError):
                self.accounting.validate_grok_prompt_accounting(candidate)

    def test_roles_and_auxiliary_coverage_stay_unknown_without_summation(self):
        result = self.project(self.ledger())
        coverage = result["coverage"]
        self.assertEqual(coverage["model_roles"], "unclassified")
        self.assertTrue(coverage["includes_prompt"])
        self.assertTrue(coverage["includes_completed_subagents"])
        self.assertFalse(coverage["includes_compaction"])
        self.assertFalse(coverage["includes_side_model_calls"])
        for key in ("primary_http_requests", "auxiliary_http_requests", "primary_tokens",
                    "auxiliary_tokens", "whole_invocation_tokens", "whole_invocation_complete"):
            self.assertIsNone(coverage[key])
        self.assertFalse(coverage["num_turns_is_http_request_count"])
        self.assertFalse(coverage["model_calls_are_http_request_count"])
        self.assertFalse(result["summation_permitted"])
        detached = self.accounting.grok_prompt_accounting_coverage()
        detached["primary_tokens"] = 999
        self.assertIsNone(self.accounting.grok_prompt_accounting_coverage()["primary_tokens"])

    def test_receipt_accepts_only_valid_nested_projection_and_detaches_it(self):
        original = self.project(self.ledger())
        result = self.accounting.grok_prompt_accounting_receipt({"native_grok_prompt_accounting": original})
        self.assertEqual(result, original)
        result["model_usage"]["rows"][0]["counters"]["input_tokens"]["value"] = 999
        self.assertEqual(original["model_usage"]["rows"][0]["counters"]["input_tokens"]["value"], 100)
        for value in (None, {}, {"native_grok_prompt_accounting": "untrusted"},
                      {"native_grok_prompt_accounting": result}):
            self.assertIsNone(self.accounting.grok_prompt_accounting_receipt(value))

    def test_native_usage_preserves_prompt_ledger_even_when_final_usage_is_absent(self):
        payload = self.ledger()
        observation = self.native_usage.native_grok_observation(payload, exit_code=0)
        self.assertEqual(observation["native_grok_prompt_accounting"], self.project(payload))
        self.assertNotIn("native_grok_usage_observed", observation)
        self.assertEqual(self.accounting.grok_prompt_accounting_receipt(observation), self.project(payload))
        # Optional accounting failure must preserve independently observed
        # native scalar usage and the original error classification.
        error_payload = {"type": "error", "subtype": "error_max_turns",
                         "num_turns": 2, "usage": {"input_tokens": 17}}
        with patch.object(self.accounting, "project_grok_prompt_accounting",
                          side_effect=RuntimeError("synthetic accounting failure")):
            failed = self.native_usage.native_grok_observation(error_payload, exit_code=1)
        self.assertNotIn("native_grok_prompt_accounting", failed)
        self.assertEqual(failed["native_grok_input_tokens"], 17)
        self.assertEqual(failed["native_grok_process_outcome"], "failed")
        self.assertEqual(failed["native_grok_reason_code"], "max_turns")
        self.assertTrue(failed["native_grok_error_observed"])

    def test_last_cli_metadata_validates_and_detaches_nested_accounting(self):
        original = self.project(self.ledger())
        self.metadata.set_last_cli_observation("grok_cli", {"exit_code": 0, "native_grok_prompt_accounting": original})
        original["num_turns"]["value"] = 99
        first = self.metadata.get_last_cli_observation("grok_cli")
        self.assertIsInstance(first["native_grok_prompt_accounting"], dict)
        self.assertEqual(first["native_grok_prompt_accounting"]["num_turns"]["value"], 3)
        self.accounting.validate_grok_prompt_accounting(first["native_grok_prompt_accounting"])
        first["native_grok_prompt_accounting"]["num_turns"]["value"] = 88
        self.assertEqual(self.metadata.get_last_cli_observation("grok_cli")["native_grok_prompt_accounting"]["num_turns"]["value"], 3)
        self.metadata.set_last_cli_observation("grok_cli", {"native_grok_prompt_accounting": original})
        self.assertNotIn("native_grok_prompt_accounting", self.metadata.get_last_cli_observation("grok_cli"))

    def test_prompt_ledger_diagnostics_leave_legacy_usage_hash_and_one_charge_unchanged(self):
        from ipfs_accelerate_py.agent_supervisor.runtime.native_session_usage_projection import reconcile_native_usage_observations

        first = self.ledger()
        first["usage"] = {"input_tokens": 7, "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1, "output_tokens": 3,
            "reasoning_tokens": 1, "total_tokens": 13}
        second = deepcopy(first)
        second.update(num_turns=50000, total_cost_usd_ticks=1 << 62,
            cost_is_partial=True, modelUsage={"unrelated-model-" + str(i): {
                "inputTokens": 1_000_000_000, "outputTokens": 200_000_000,
                "modelCalls": 10000} for i in range(40)})
        observed_first = self.native_usage.native_grok_observation(first, exit_code=0)
        observed_second = self.native_usage.native_grok_observation(second, exit_code=0)
        self.assertNotEqual(observed_first["native_grok_prompt_accounting"],
                            observed_second["native_grok_prompt_accounting"])
        self.assertEqual(observed_first["native_grok_usage_sha256"], observed_second["native_grok_usage_sha256"])
        usage = self.native_usage.grok_usage_receipt(observed_first)
        self.assertEqual(usage, self.native_usage.grok_usage_receipt(observed_second))
        self.assertTrue(usage["usage_complete_observed"])
        outcome = self.native_usage.grok_outcome_receipt(observed_first)
        self.assertEqual(outcome, self.native_usage.grok_outcome_receipt(observed_second))
        wrapper = {"schema": "router-implementation-invocation@1",
            "invocation_id": "unit-ledger-sibling-invocation", "provider": "grok_cli",
            "model": "grok-4.7", "workspace": str(self.root), "status": "provider_returned",
            "usage": {"session_id": "unit-ledger-sibling-session", "model_id": "grok-4.7"},
            "native_rollout_usage": usage, "native_provider_outcome": outcome,
            "completion_authority": False}
        baseline = reconcile_native_usage_observations([wrapper, deepcopy(wrapper)])
        supplied = [deepcopy(wrapper), deepcopy(wrapper)]
        supplied[0]["native_prompt_accounting"] = self.accounting.grok_prompt_accounting_receipt(observed_first)
        supplied[1]["native_prompt_accounting"] = self.accounting.grok_prompt_accounting_receipt(observed_second)
        original = deepcopy(supplied)
        result = reconcile_native_usage_observations(supplied)
        self.assertEqual(supplied, original)
        self.assertEqual(result, baseline)
        self.assertEqual(result["duplicate_wrapper_observations"], 1)
        bucket = result["grok_final_envelopes"]
        self.assertEqual(len(bucket["rows"]), 1)
        self.assertTrue(bucket["diagnostic_subtotal_available"])
        self.assertEqual(bucket["diagnostic_subtotals"], {
            "input_tokens": 7, "cached_input_tokens": 2, "cache_write_input_tokens": 1,
            "output_tokens": 3, "reasoning_output_tokens": 1, "total_tokens": 13,
            "normalized_input_tokens": 10, "normalized_total_tokens": 13})
        self.assertTrue(bucket["rows"][0]["observed_native_scope_complete"])
        self.assertIsNone(bucket["coverage"]["primary_http_requests"])
        self.assertIsNone(bucket["coverage"]["auxiliary_http_requests"])
        self.assertIsNone(bucket["coverage"]["auxiliary_tokens"])
        self.assertIsNone(bucket["coverage"]["whole_invocation_tokens"])
        self.assertFalse(result["billing_total_verified"])
        self.assertFalse(result["completion_authority"])

    def runner_call(self, failure=None):
        from ipfs_accelerate_py.llm_allocation import intelligence_index
        from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting

        payload = self.ledger()
        receipts = []
        expected = self.project(payload)

        def generate(_prompt, **kwargs):
            self.assertFalse(kwargs["allow_local_fallback"])
            self.assertFalse(kwargs["allow_cross_provider_fallback"])
            observation = self.native_usage.native_grok_observation(payload, exit_code=1 if failure else 0)
            self.metadata.set_last_cli_observation("grok_cli", {"exit_code": 1 if failure else 0, **observation})
            if failure is not None:
                raise failure
            return "synthetic runner response"

        with (patch.object(ctypes.util, "find_library", side_effect=_fixture_library_resolver()),
              patch.object(subprocess, "check_output", return_value=str(self.root)),
              contextlib.chdir(self.root),
              patch.object(intelligence_index, "discover_available_providers", return_value=["grok_cli"]),
              patch.object(intelligence_index, "select_efficient_route", return_value=SimpleNamespace(
                  provider="grok_cli", model_name="grok-4.7", catalog_revision="unit-only")),
              patch.object(self.router, "get_llm_provider", return_value=object()),
              patch.object(self.router, "generate_text", side_effect=generate) as transport,
              patch.object(context_delivery_accounting, "observe_invocation", return_value={"status": "unit-only"}),
              contextlib.redirect_stdout(io.StringIO())):
            if failure is None:
                output, receipt = self.runner.run(prompt="synthetic runner instruction", provider="grok_cli",
                    model="grok-4.7", purpose="planning", timeout=30, max_output_tokens=4096,
                    receipt_observer=receipts.append)
                self.assertEqual(output, "synthetic runner response")
                self.assertEqual(receipt["native_prompt_accounting"], expected)
            else:
                with self.assertRaises(type(failure)) as caught:
                    self.runner.run(prompt="synthetic runner instruction", provider="grok_cli",
                        model="grok-4.7", purpose="planning", timeout=30, max_output_tokens=4096,
                        receipt_observer=receipts.append)
                self.assertIs(caught.exception, failure)
            self.assertEqual(transport.call_count, 1)
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["native_prompt_accounting"], expected)
        self.assertFalse(receipts[0]["completion_authority"])
        self.assertFalse(receipts[0]["native_prompt_accounting"]["billing_total_verified"])
        return receipts[0]

    def test_actual_runner_success_propagates_validated_prompt_accounting(self):
        receipt = self.runner_call()
        self.assertEqual(receipt["status"], "provider_returned")

    def test_actual_runner_failure_retains_accounting_and_original_exception(self):
        failure = RuntimeError("synthetic provider failure")
        receipt = self.runner_call(failure)
        self.assertEqual(receipt["native_prompt_accounting"]["num_turns"]["value"], 3)


if __name__ == "__main__":
    unittest.main()
