import hashlib,json,subprocess,time
from pathlib import Path
b=Path(__file__).parent
name="torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
expected="6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604"
url="https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl"
final=b/name;partial=b/(name+".partial")
assert not final.exists()
argv=["curl","--fail","--location","--proto","=https","--user-agent","pip/25.3","--connect-timeout","30","--max-time","600","--max-filesize","155005253","--continue-at","-","--output",str(partial),url]
(b/"command.json").write_text(json.dumps(dict(argv=argv,expected_sha256=expected,expected_bytes=155005253),indent=2)+"\n")
start=time.monotonic()
with (b/"indexed-download.log").open("a") as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
receipt=dict(returncode=r.returncode,seconds=time.monotonic()-start,bytes=partial.stat().st_size if partial.exists() else 0,url=url)
if r.returncode==0:
 assert partial.stat().st_size==155005253
 digest=hashlib.file_digest(partial.open("rb"),"sha256").hexdigest()
 assert digest==expected
 partial.rename(final);receipt.update(sha256=digest,verified=True,path=str(final.resolve()))
(b/"download-result.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt));raise SystemExit(r.returncode)
