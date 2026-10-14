"""Copy only reviewed pure diagnostic validators; never import runtime modules."""
import ast
import hashlib
import json
from pathlib import Path
A=Path(__file__).resolve().parent
W=Path('/home/barberb/lift_coding/.worktrees/supervisor-doctor-live-fixes-20261014')
class LocalImports(ast.NodeTransformer):
 def visit_ImportFrom(self,node):
  if node.module == 'runtime.router_implementation_runner':return ast.ImportFrom(module='diagnostic_runner_contract',names=node.names,level=0)
  if node.module == 'native_provider_custody_observation':return ast.ImportFrom(module='diagnostic_custody_contract',names=node.names,level=0)
  return node
records=[]
def extract(relative,target,functions,constants=None,imports=''):
 source=W/relative;tree=ast.parse(source.read_text());selected=[]
 for node in tree.body:
  if isinstance(node,ast.Assign):
   names=[t.id for t in node.targets if isinstance(t,ast.Name)]
   if constants is None or any(n in constants for n in names):
    if names==['_FRAME_NAMES']:
     mapping=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FRAME_PATHS' for t in n.targets))
     files=ast.literal_eval(mapping.value.generators[0].iter)
     node=ast.parse('_FRAME_NAMES=frozenset('+repr(sorted(Path(p).name for p in files))+')').body[0]
    selected.append(node)
  elif isinstance(node,ast.FunctionDef) and node.name in functions:selected.append(LocalImports().visit(node))
 out=A/target;out.write_text('"""Frozen pure validators extracted from qualified source; no runtime imports or I/O."""\nfrom __future__ import annotations\n'+imports+'\n'+ast.unparse(ast.Module(body=selected,type_ignores=[]))+'\n')
 records.append(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=str(out),output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),selected_functions=sorted(functions),selected_constants=sorted(constants) if constants else 'all top-level literal assignments'))
extract('ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py','diagnostic_runner_contract.py',{'validate_runner_failure_diagnostic','validate_runner_error_envelope'},{'RUNNER_FAILURE_PHASES','RUNNER_FAILURE_TYPES','RUNNER_FAILURE_FILES'},'import json')
extract('ipfs_accelerate_py/agent_supervisor/todo_daemon/native_provider_custody_observation.py','diagnostic_custody_contract.py',{'validate_native_provider_custody_observation'})
extract('ipfs_accelerate_py/agent_supervisor/todo_daemon/bridge_failure_diagnostics.py','diagnostic_bridge_contract.py',{'_returncode','missing_child_report','validate_child_report','_diagnostic_key','_validate_multiple_child_report','validate_bridge_failure_diagnostic'}, {'SCHEMA','LEGACY_SCHEMA','MULTIPLE_CHILD_SCHEMA','MAX_CHILD_DIAGNOSTICS','_MULTIPLE_CHILD_FLAGS','_TYPES','_REASONS','_CALLBACK_SCHEMAS','_STATES','_EFFECTS','_EXIT_BOOLS','_FRAME_NAMES'},'import json')
extract('ipfs_accelerate_py/cli_runtime/grok_stream_usage.py','diagnostic_grok_usage_contract.py',{'_count','validate_grok_usage_events'},None,'import math')
extract('ipfs_accelerate_py/cli_runtime/grok_stream_capture.py','diagnostic_grok_progress_contract.py',{'_number','validate_grok_stream_progress','grok_stream_progress_summary'},None,'import math\nfrom diagnostic_grok_usage_contract import validate_grok_usage_events')
(A/'reader-contract-provenance.json').write_text(json.dumps({'schema':'pure-diagnostic-reader-contracts@1','runtime_imports':False,'records':records},indent=2,sort_keys=True)+'\n')
