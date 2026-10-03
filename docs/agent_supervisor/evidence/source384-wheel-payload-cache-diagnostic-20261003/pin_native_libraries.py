"""Reproduce finite source pins without copying wheel members or models."""
import hashlib,json,stat,sys,zipfile
from pathlib import Path
MANIFEST='4d13cfeb3ad2daf67a50c135babbf58c6a197a71f70376bc730051508ce8fb27'
WHEEL='6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604'
PREFIXES=('torch/','torchgen/','functorch/')
def digest(stream):
 h=hashlib.sha256()
 for raw in iter(lambda:stream.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
def main(manifest,wheel):
 if manifest.stat().st_size>8*1024**2 or wheel.stat().st_size>256*1024**2:raise ValueError('source bound exceeded')
 raw=manifest.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=MANIFEST:raise ValueError('manifest digest differs')
 with wheel.open('rb') as f:
  if digest(f)!=WHEEL:raise ValueError('official wheel digest differs')
 value=json.loads(raw);ext=[r for r in value['files'] if r['path'].startswith('extensions/')]
 if {r['path'] for r in ext}!={'extensions/ducklake.duckdb_extension','extensions/httpfs.duckdb_extension','extensions/quack.duckdb_extension'} or len(ext)!=3:raise ValueError('extension population differs')
 rows=[]
 with zipfile.ZipFile(wheel) as archive:
  selected=sorted((i for i in archive.infolist() if i.filename.startswith(PREFIXES) and not i.is_dir()),key=lambda i:(-i.file_size,i.filename))[:128]
  if len(selected)!=128:raise ValueError('exact128-member selection required')
  for chosen in selected:
   name=chosen.filename
   found=[i for i in archive.infolist() if i.filename==name]
   if len(found)!=1:raise ValueError('unique wheel member required')
   item=found[0]
   if not stat.S_ISREG(item.external_attr>>16) or not 0<item.file_size<=512*1024**2:raise ValueError('bounded regular member required')
   with archive.open(item) as stream:sha=digest(stream)
   rows.append(dict(wheel_member=name,bytes=item.file_size,sha256=sha,zip_mode=stat.S_IMODE(item.external_attr>>16),mode=0o755 if (item.external_attr>>16)&0o111 else 0o644))
 print(json.dumps(dict(schema='finite-native-library-source-pins@1',archive_manifest_sha256=MANIFEST,wheel_sha256=WHEEL,extensions=ext,torch=rows,wheel_members_read=128,selection='descending_uncompressed_bytes_then_path_top128_torch_torchgen_functorch',installer_mode_policy='pip_normal_file_0666_umask022_with_executable0755',provider_calls=0),indent=2,sort_keys=True))
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
