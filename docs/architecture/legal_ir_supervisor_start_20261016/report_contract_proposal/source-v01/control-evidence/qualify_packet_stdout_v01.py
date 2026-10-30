"""Effect-free packet controls with genuine native CPU ownership and stdout."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,unittest
P=Path(__file__).absolute().parent
D=Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
def main():
 path=Path(sys.argv[1]);raw=path.read_bytes();r=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
 def guard():
  assert path.read_bytes()==raw
  for f,h in r['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 guard();sys.dont_write_bytecode=True
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',IPFS_DATASETS_PY_MINIMAL_IMPORTS='1',IPFS_AUTO_INSTALL='false',IPFS_DATASETS_AUTO_INSTALL='false',IPFS_ACCEL_SKIP_CORE='1')
 sys.path.insert(0,str(D))
 from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
 scheduler=open_existing_scheduler('/tmp/ipfs-datasets-resource-scheduler-1000.json','f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661')
 with scheduler.acquire('orchestration',cpu_slots=1,memory_mb=1024,child_process_slots=1,requires_gpu=False,timeout=60,request_id=r['name']+':root') as root:
  with root.acquire_child(cpu_slots=1,memory_mb=1024,child_process_slots=1,requires_gpu=False,timeout=60,request_id=r['name']+':phase') as phase:
   guard();test=P/'real-report-packet-v01/test_packet_contract_memory.py'
   spec=importlib.util.spec_from_file_location('qualified_packet_contract_tests',test);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
   suite=unittest.defaultTestLoader.loadTestsFromModule(m);assert suite.countTestCases()==21
   result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
   guard();assert root.renew() and phase.renew()
 print(json.dumps({'schema':'legal-ir-packet-controls-native-stdout-terminal@1','passed':result.wasSuccessful(),'controlled_testcases':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_request_sha256':hashlib.sha256(raw).hexdigest(),'actual_root_released':root.released,'actual_phase_released':phase.released,'filesystem_test_outputs_requested':False,'native_disk_grant_claimed':False,'packet_materialized':False,'provider_calls_performed':False,'canonical_tasks_accepted':0}),flush=True)
 return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
