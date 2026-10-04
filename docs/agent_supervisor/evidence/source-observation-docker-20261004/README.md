# Bounded source observation passes ordinary Docker qualification

Archive 73fe0fa56e700075f9198ab677c7a0b4b8be74e4971500f81b15c6476d089241
uses accelerate a42d2c8a60b0bc0b4345dc91228cd8b721d49008 and datasets
88d4016091284cff5d0eb3afcccf82b9934affd7. Its manifest is
5841a2f5c2ecc0b7a72e36215ea2c3052e7b3b483e2a76e381cc4b96de5f4a18.
All 26929 members were checked for content hash, size, mode and ownership.
Four runtime owners changed from the preceding failed archive. One upstream
scalar diagnostic module was added; existing tested producers remain unchanged.
Checkpoint, GTE, solvers, reviewed intent, resources and deadlines are unchanged.

Ordinary production qualification passes with no diagnostic hooks. Deployment
takes 203.698s, source preparation 11.468s, initial context 143.801s, warm source
observation 10.988s and the native probe 167.404s. Source384 takes 83.774s within
its 90-second deadline. The controller takes 440.094s. These are nested timing
scopes, not additive phases. Source/task pins remain unchanged and container
cleanup is verified. This single run has narrow deadline headroom and does not
establish causal or general performance improvement.

The actual unchanged 384-dimensional checkpoint is consumed at inference time.
Coverage remains 31 Python files, 944 inventoried functions and 128 selected
units: 127 decoded unverified candidates and one GTE token-limit deferral.
The 127 candidates remain unsupported by the source-contract gate; the other
functions retain explicit selection or normalization dispositions. No learned
formula is promoted to a proof. There are zero provider calls and training
steps. The signed reviewed header selector is administrative input, not a
learned IntentIR semantic proof; its review cost is unavailable and excluded.

This package qualifies the prerequisite only. No official verifier is run here,
and it supplies no task reward, completed token score or matched-arm advantage.
The subsequent full trial is retained separately. Exports contain reviewed
runtime metadata only, with raw model, prompt, verifier and credential bodies
excluded.
