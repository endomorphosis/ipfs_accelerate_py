import json
import asyncio
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import resource_runtime as r

class ResourceInstrumentationTests(unittest.TestCase):
    def test_native_producer_recognized_before_after_and_restored(self):
        from ipfs_datasets_py.logic.software_contracts import codebase_ir as owner
        from ipfs_datasets_py.logic.software_contracts import codebase_source_units_384 as units
        original=(owner.RepositoryCodebaseIndex.prepare_current,owner.RepositoryCodebaseIndex.observe_current,units._worker)
        key=owner._manifest_producer_key();self.assertIsNotNone(key)
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'_RESOURCE_PATH',str(Path(directory)/'events')):
            try:
                r._resource_install()
                r._resource_wrap(units,'_worker','numerical_worker')
                self.assertEqual(owner._manifest_producer_key(),key)
                self.assertEqual(r._resource_state['events'],1)
            finally:r._resource_finish()
            self.assertEqual(original,(owner.RepositoryCodebaseIndex.prepare_current,owner.RepositoryCodebaseIndex.observe_current,units._worker))
            events=[json.loads(line) for line in Path(r._RESOURCE_PATH).read_text().splitlines()]
            self.assertEqual(len(events),2)
            self.assertEqual(set(events[0]),{'stage','event','error_type','elapsed_seconds','host','memory_current_bytes','memory_stat','process_statm'})
            self.assertNotIn('state_path',events[0]);self.assertNotIn('source',events[0])
        r._resource_state=None

    def test_alarm_exception_is_not_swallowed_by_optional_collection(self):
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import proof_resource_safety as safety
        r._resource_state={'events':0,'started':0.,'collection_errors':0}
        with patch.object(safety,'collect_proof_host_resources',side_effect=TimeoutError('native deadline')):
            with self.assertRaises(TimeoutError):r._resource_emit('worker','return')
        r._resource_state=None

    def test_exact_probe_preserves_native_budgets_and_single_splice(self):
        from benchmarks.agent_supervisor.container_coding.terminal_source384_qualification import CONTEXT_PROBE
        from run_diagnostic import instrument_probe
        value=instrument_probe(CONTEXT_PROBE)
        self.assertEqual(value.count('signal.setitimer(signal.ITIMER_REAL,270)'),1)
        self.assertIn('  _resource_install()\n',value)
        self.assertIn("result['production_qualification_claimed']=False",value)
        with self.assertRaises(ValueError):instrument_probe(CONTEXT_PROBE.replace("  result['producer']=_pins()\n",''))

    def test_actual_splice_final_alarm_propagates_and_disarms(self):
        import run_diagnostic as wrapper
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'resource_runtime.py').write_text("def _resource_install(): pass\ndef _resource_finish(): raise TimeoutError('native alarm in optional finish')\n")
            original="result={}\nsignal.setitimer(signal.ITIMER_REAL,270)\ntry:\n if True:\n  result['producer']=_pins()\nfinally:\n signal.setitimer(signal.ITIMER_REAL,0)\n"
            calls=[];namespace=dict(signal=SimpleNamespace(ITIMER_REAL=0,setitimer=lambda *args:calls.append(args)),_pins=lambda:{})
            with patch.object(wrapper,'BASE',base):code=wrapper.instrument_probe(original)
            with self.assertRaises(TimeoutError):exec(code,namespace)
            self.assertEqual(calls,[(0,270),(0,0)])

    def test_splice_or_receipt_write_failure_never_installs_hooks(self):
        import run_diagnostic as wrapper
        from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
        from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualification
        before=(deployment.deploy_supervisor,qualification.qualify_context,qualification.observe_resources,qualification.CONTEXT_PROBE)
        with patch.object(wrapper,'instrument_probe',side_effect=ValueError('wrong native anchor')):
            with self.assertRaises(ValueError):asyncio.run(wrapper.main())
        self.assertEqual(before,(deployment.deploy_supervisor,qualification.qualify_context,qualification.observe_resources,qualification.CONTEXT_PROBE))
        with patch.object(Path,'write_text',side_effect=OSError('cannot retain instrumented probe')):
            with self.assertRaises(OSError):asyncio.run(wrapper.main())
        self.assertEqual(before,(deployment.deploy_supervisor,qualification.qualify_context,qualification.observe_resources,qualification.CONTEXT_PROBE))

if __name__=='__main__':unittest.main()
