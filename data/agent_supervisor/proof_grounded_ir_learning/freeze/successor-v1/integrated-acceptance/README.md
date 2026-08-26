# PGIR-211 integrated acceptance

This directory records the non-circular integrated replay of the immutable
PGIR-208 through PGIR-210 target forest. Materialize evidence in this order:
`historical`, `tests`, `network`, `portability`, `component`, then `acceptance`.
Every Python stage and verifier invocation uses `/usr/bin/python3.12 -S` with
the exact raw shared-package roots and startup controls enforced by the
scripts. The network stage captures all 21 exact-revision HTTPS responses;
the final `--network` verification performs 21 live requests again and cannot
substitute the frozen receipt.

The portability capture binds current advertised heads and tags from fresh
isolated bare repositories. The integrated outer target and all required
nested commits are currently reachable. Portability remains a typed no-go
only because outer closure authorities
`04fbb09b4a8b34e77d11bd8da6642e0978baa02c` and
`597a0285738c5878eed462593fd75e18715ff7f8` are not reachable from any fetched
advertised head or tag. The fresh-recursive-checkout blocker names the first
of those actually missing commits; it does not claim that the target is
unreachable. Any later remote-state change fails the exact capture closed.

`verified=true` means the replay, test, portability, and live-network claims
were reproduced. It does not mean the historical inputs are eligible for
learning: the canonical decision remains `permanent_no_go`, completion is
non-authoritative, and PGIR-205 execution remains prohibited. Offline/plain
verification fails closed.
