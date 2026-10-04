from pathlib import Path
import json,time
B=Path(__file__).resolve().parent
started=time.monotonic();count=0
with (B/'pressure.jsonl').open('x') as stream:
 while time.monotonic()-started<4500:
  memory={}
  for line in Path('/proc/meminfo').read_text().splitlines():
   k,v=line.split(':',1)
   if k in {'MemAvailable','MemFree','SwapFree'}:memory[k+'_kib']=int(v.split()[0])
  psi={}
  for line in Path('/proc/pressure/memory').read_text().splitlines():
   fields=line.split();values=dict(f.split('=') for f in fields[1:]);psi[fields[0]]={k:float(values[k]) for k in ['avg10','avg60','avg300']}
  service=json.loads((B/'service-pause.json').read_text())
  row={'at':time.time(),'elapsed_seconds':time.monotonic()-started,'phase':service['events'][-1]['phase'],'memory':memory,'memory_psi_percent':psi}
  stream.write(json.dumps(row)+'\n');stream.flush();count+=1
  if 'returncode' in service:break
  time.sleep(5)
(B/'pressure-exit.json').write_text(json.dumps({'samples':count,'elapsed_seconds':time.monotonic()-started,'observational_only':True})+'\n')
