"""Lightweight wrapper-contract controls. No repositories, models or Docker run."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

B=Path(__file__).parent


def load(name):
    spec=importlib.util.spec_from_file_location(name,B/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiagnosticControls(unittest.TestCase):
    def setUp(self):
        from ipfs_datasets_py.logic.software_contracts import codebase_ir, duckdb_ast_store
        self.index=codebase_ir
        self.store=duckdb_ast_store
        self.temp=tempfile.TemporaryDirectory()
        self.runtime=load('timing_runtime')
        self.runtime._DIAG_EVENTS_PATH=str(Path(self.temp.name)/'events.jsonl')
        self.runtime._DIAG_SUMMARY_PATH=str(Path(self.temp.name)/'summary.json')

    def tearDown(self):
        state=self.runtime._diag_state
        if state is not None and not state['stream'].closed:
            self.runtime._diag_finish()
        self.temp.cleanup()

    def test_native_guard_survives_wrappers_and_units_are_lazy(self):
        m=self.runtime
        original=self.index.RepositoryCodebaseIndex.prepare_current
        original_static=self.store.DuckDBASTStore.__dict__['_rebuild']
        key=self.index._manifest_producer_key()
        self.assertIsNotNone(key)
        before_units=sys.modules.get('ipfs_datasets_py.logic.software_contracts.codebase_source_units_384')
        m._diag_install()
        self.assertIs(sys.modules.get('ipfs_datasets_py.logic.software_contracts.codebase_source_units_384'),before_units)
        self.assertEqual(self.index._manifest_producer_key(),key)
        self.assertFalse(m._diag_state['units_installed'])
        # Test-only import simulates the production import inside its native timer.
        from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384
        original_worker=codebase_source_units_384._worker
        m._diag_install_units()
        self.assertEqual(self.index._manifest_producer_key(),key)
        self.assertIsNot(codebase_source_units_384._worker,original_worker)
        # Tiny explicit control only, never benchmark source or model input.
        from ipfs_datasets_py.logic.software_contracts import content
        for _ in range(2):
            content.validate_cid(content.cid_for_bytes(b'diagnostic-control-only'))
        m._diag_finish()
        self.assertIs(codebase_source_units_384._worker,original_worker)
        self.assertIs(self.index.RepositoryCodebaseIndex.prepare_current,original)
        self.assertIs(self.store.DuckDBASTStore.__dict__['_rebuild'],original_static)
        summary=json.loads(Path(m._DIAG_SUMMARY_PATH).read_text())
        (B/'controls-native-guard.json').write_text(json.dumps(summary,indent=2)+'\n')
        self.assertTrue(summary['after_context']['manifest_producer_unchanged'])
        self.assertTrue(summary['after_context']['cid_preexisting_cached_layout_recognized'])
        self.assertGreater(summary['after_context']['cid_encode_memo']['hits'],
            summary['before_wrappers']['cid_encode_memo']['hits'])
        self.assertGreater(summary['after_context']['cid_validate_memo']['hits'],
            summary['before_wrappers']['cid_validate_memo']['hits'])
        self.assertEqual(set(summary['after_context']['manifest_memo']),
            {'hits','misses','evictions','bypasses','entries','retained_bytes'})

    def test_timeout_arguments_staticmethod_and_exceptions_preserved_without_payloads(self):
        m=self.runtime;m._diag_install()
        calls=[]
        sentinel=ValueError('SECRET_EXCEPTION_CONTENT')
        class Sample:
            @staticmethod
            def rebuild(payload,*,timeout,memory_mb):
                calls.append((payload,timeout,memory_mb))
                return timeout
            def fail(self,secret,*,timeout_seconds):
                raise sentinel
        m._diag_wrap(Sample,'rebuild','test.static')
        m._diag_wrap(Sample,'fail','test.failure')
        self.assertEqual(Sample().rebuild('SECRET_SOURCE',timeout=7.25,memory_mb=4096),7.25)
        self.assertEqual(calls,[('SECRET_SOURCE',7.25,4096)])
        with self.assertRaises(ValueError) as caught:
            Sample().fail('SECRET_FORMULA',timeout_seconds=3.125)
        self.assertIs(caught.exception,sentinel)
        m._diag_finish()
        text=Path(m._DIAG_EVENTS_PATH).read_text()
        self.assertNotIn('SECRET',text)
        events=[json.loads(line) for line in text.splitlines()]
        self.assertTrue(any(row.get('timeout')==7.25 and row.get('memory_mb')==4096 for row in events))
        self.assertTrue(any(row.get('error_type')=='ValueError' for row in events))

    def test_snapshots_do_not_initialize_native_layout_or_cid_memos(self):
        from ipfs_datasets_py.logic.software_contracts import content
        content._native_registry_layout.cache_clear()
        before=[fn.cache_info() for fn in (content._native_registry_layout,
            content._memo_encode_digest,content._memo_validate_cid)]
        self.runtime._diag_install()
        self.runtime._diag_snapshot()
        self.runtime._diag_finish()
        after=[fn.cache_info() for fn in (content._native_registry_layout,
            content._memo_encode_digest,content._memo_validate_cid)]
        self.assertEqual(after,before)

    def test_event_count_limit_records_drops(self):
        m=self.runtime;m._DIAG_MAX_EVENTS=3;m._diag_install()
        for _ in range(50):m._diag_emit({'event':'bounded_test'})
        m._diag_finish()
        summary=json.loads(Path(m._DIAG_SUMMARY_PATH).read_text())
        self.assertEqual(summary['events'],3)
        self.assertEqual(summary['dropped_events'],48)

    def test_restore_wrappers_even_when_diagnostic_stream_close_fails(self):
        m=self.runtime
        original=self.index.RepositoryCodebaseIndex.prepare_current
        m._diag_install()
        stream=m._diag_state['stream']
        class BrokenClose:
            def __getattr__(self,name):return getattr(stream,name)
            def close(self):
                stream.close()
                raise OSError('diagnostic close failure')
        m._diag_state['stream']=BrokenClose()
        with self.assertRaises(OSError):m._diag_finish()
        self.assertIs(self.index.RepositoryCodebaseIndex.prepare_current,original)

    def test_event_byte_limit_records_drops(self):
        m=self.runtime;m._DIAG_MAX_BYTES=128;m._diag_install()
        m._diag_emit({'event':'x'*1024})
        m._diag_finish()
        summary=json.loads(Path(m._DIAG_SUMMARY_PATH).read_text())
        self.assertLessEqual(Path(m._DIAG_EVENTS_PATH).stat().st_size,128)
        self.assertGreater(summary['dropped_events'],0)

    def test_probe_injection_keeps_original_runtime_and_alarm(self):
        from benchmarks.agent_supervisor.container_coding.terminal_source384_qualification import CONTEXT_PROBE
        runner=load('run_diagnostic')
        output=runner.instrument_probe(CONTEXT_PROBE)
        compile(output,'<diagnostic-control>','exec')
        self.assertEqual(output.count('signal.setitimer(signal.ITIMER_REAL,270)'),1)
        self.assertIn('context=prep.initial_context(state=state,source384_config=config_path,train_autoencoder=False)',output)
        self.assertIn("result['production_qualification_claimed']=False",output)


if __name__=='__main__':
    unittest.main(verbosity=2)
