"""Condense only an already qualified, closed-schema readout."""
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
rows={}
for side in ('prior','pilot'):
 b=r['baseline'].get(side)
 if b is None:continue
 p=r['planning'][side]['trials'][0]['planning']
 for t in b['trials']:
  inv=[]
  for i in t['invocations']:
   progress=i.get('native_grok_progress',{});state=progress.get('progress',{})
   inv.append(dict(phase=i.get('phase'),status=i.get('status'),failure_phase=i.get('failure_phase'),native_cli=i.get('native_cli'),usage=i.get('native_usage'),model_prompt_bytes=i.get('model_prompt_bytes'),native_progress={k:state.get(k) for k in ('final_event_type','final_reason_code','stream_tainted','returncode','duration_seconds','events_after_final')},terminal_rejection=progress.get('terminal_rejection')))
  rows[side]=dict(trial=t['trial'],official_reward=t['official_reward'],task_completed_reported=t.get('task_completed_reported'),seconds=t['seconds'],phase_seconds=t['phase_seconds'],planning=p,all_phase_usage=t['all_phase_usage'],invocations=inv,continuation=t.get('continuation'),public_validation={k:t.get('public_validation',{}).get(k) for k in ('status','owner_validation_outcome','owner_returncode')},rounds=[dict(round_number=v['round_number'],actual_returncode=v['actual_returncode'],status=v['status'],workspace_generation=v['workspace_generation'],checks=[{k:c[k] for k in ('validation_key','status','reason')} for c in v['checks']]) for v in t.get('public_validation',{}).get('rounds',[])])
print(json.dumps(rows,indent=2,sort_keys=True))
