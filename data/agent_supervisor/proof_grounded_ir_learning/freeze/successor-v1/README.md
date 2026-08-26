# PGIR-205 successor-v1 campaign input freeze

This directory is the write-once successor location for `RESULT(PGIR-205)`.
It does not modify the historical PGIR-014 freeze under
`data/agent_supervisor/proof_grounded_ir_learning/freeze/`.

`previous_root_cid` is the historical no-go root
`baguqeerarkgpz4xl663tlpfpiajjtxlya3b576lqzg5yd7nrthqgs2rm6v2q`.
The freeze binds `RESULT(PGIR-211)`, the later PGIR-211 implementation /
merge / completion forest that `RESULT(PGIR-211)` itself cannot name, and a
fresh run of
`scripts/verify_proof_grounded_ir_learning_successor_integrated_acceptance.py`
at this baseline.

Learned descendant tasks are eligible only when every gate passes: rights,
corpus, holdouts, tokenizer, current-baseline, integrated-evidence, and
portability. All seven currently fail. Portability is a typed
`portability_no_go` (unpublished outer refs, including the later PGIR-211
forest and the historical closure authorities already recorded by
PGIR-211). The campaign decision is therefore a documented `no_go`:
integrity success, execution denial. PGIR-206 must emit typed `not_run`
evidence rather than train.

Predecessor evidence already sealed in this tree remains an input:

- `tokenizer/` — `RESULT(PGIR-203)`
- `source-chain-acceptance/` — `RESULT(PGIR-208)`
- `baseline-acceptance/` — `RESULT(PGIR-209)`
- `source-chain-acceptance-v2/` — `RESULT(PGIR-210)`
- `integrated-acceptance/` — `RESULT(PGIR-211)`

Run the independent verifier from the repository root in a raw, isolated,
no-site Python environment:

```bash
env -i \
  GIT_CONFIG_COUNT=0 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_TERMINAL_PROMPT=0 LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/barberb/.local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TZ=UTC \
  /usr/bin/python3.12 -S \
  data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/verify_campaign_freeze.py
```

That default is offline: it independently replays all immutable identities,
Git and file bindings, fail-closed gates, and the sealed strict-network run
receipt without contacting any provider. To require a new live 21-object
HTTPS replay as well, append `--fresh-network` to the command. There is no
component-receipt fallback for that mode.

Builder replay is also read-only and offline by default:

```bash
env -i \
  GIT_CONFIG_COUNT=0 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_TERMINAL_PROMPT=0 LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/barberb/.local/lib/python3.12/site-packages:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TZ=UTC \
  /usr/bin/python3.12 -S \
  data/agent_supervisor/proof_grounded_ir_learning/freeze/successor-v1/build_campaign_freeze.py
```

On first issuance only, append `--initialize`; it captures the strict network
and portability observations only when both sealed capture files are absent,
creates absent artifacts, and refuses to replace different bytes. Subsequent
default builder and verifier runs consume those sealed captures and perform no
network access. Any admitted replacement must use a new task revision and a
separately located root whose `previous_root_cid` points here.
