# AST text validation and reconciled consumer controls

Four fresh selected groups pass **756 unique test identities**, with zero
failures, errors or skips: 224 supervisor, 142 integration, 153 serialization/
identity controls and 237 AST text/frontend/store controls. Their commands,
exits and JUnit reports are retained. Each run reports unchanged source/test
pins, and the passing groups share consistent exact hashes. Their source
generation is the reconciled base, accelerate
`d2de765bc5307cd728a9c9099759e812ee092381` and datasets
`da853fa837b89a4ef01fa981127cfc446158371e`, plus four explicitly retained tested
accelerate source/test overlays. Other selected pins match their base commits.
These are component results, not complete native or Docker qualification.

The exact-string AST validator replaces a per-character printable check with
the built-in whole-string check, preserving the allowed-empty case and original
type/whitespace/length/NFC/error order. Its 148 new controls include one loop over
all 1,114,112 Unicode code points. The previously sealed datasets performance
package is referenced in summary.json by commit/path/manifest rather than copied:
it contains the original 28b4-based public Bottle observations, not measurements
on the reconciled source. Those small observations show lower AST reconstruction
CPU but only a small reopened-store change and a persistence regression; they
establish no full-publication, native-task, RSS or token-efficiency improvement.

The native source-unit family **did not pass**. Pre-reconciliation native-01
has six passing preparation/tamper cases and one actual checkpoint inference/
replay case failing with LeaseTimeoutError during source observation. On the
reconciled generation, native-02 has seven setup errors: the shared source
capture fixture fails resource admission before any test body can complete.
Both exact generations and failed reports remain explicit; the six earlier
passes are not added to the 756 current passing identities. There is no complete
native-family pass claim, successful native inference-family qualification or
new Docker result.

A later independent bounded readiness request also refuses memory pressure.
Its request-local sample records aggregate memory full avg10 **8.88%**, host
**6.01%** and a visible cgroup ancestor **8.88%**, against the unchanged **2%**
limit. Available memory is 41,818 MiB, above 24,922 MiB headroom plus the 512 MiB
request, with no root reservations. This identifies that later request's gate;
it does not establish the cause of either preceding native failure. The
readiness result is not a native test, inference run or backdated observation.

After the passing controls, another independent readiness request still
refuses admission: aggregate/ancestor memory full avg10 is **18.83%**, host
**14.41%**, against the same **2%** limit. Its 41,100 MiB available memory
exceeds 24,922 MiB headroom plus the 512 MiB request. This final separate
observation does not explain the earlier failures. No native-03 test run follows
that refusal, and the seven-case live family remains unqualified.

The reconciliation preserves upstream work and introduces additional runtime
changes beyond the one local AST condition. Its observation explicitly retires
the preceding one-file archive-delta declaration. A fresh complete archive
review and ordinary qualification remain necessary. The final 224-case consumer
group includes the restored admitted runtime's per-run orchestration binding
and its 11 integration controls, plus the new resource-diagnostic PID-refusal
case. The preceding source-03 run passed 223 cases and failed that new case at
fixture construction because it supplied PID availability without the required
paired limit. The fixture was corrected and the complete 224-case suite rerun;
the failed report remains separate and is not counted as qualifying evidence.
The restored orchestration and bounded diagnostic owners retain their own
review scope; these component passes do not repair the absent native inference
qualification or establish a new admitted Docker/full-task result.
No fresh Docker archive or qualification has been produced after this broad
upstream reconciliation.

The backlog remains 18 of 32 closed. No current-source successor, learned proof
authority, completed token score or matched-arm advantage is established.
Only bounded metadata, test reports and diagnostic recipes are exported. Logs
listed as local references are not package members; private stores, model
weights, credentials and hidden verifier bodies are excluded.
