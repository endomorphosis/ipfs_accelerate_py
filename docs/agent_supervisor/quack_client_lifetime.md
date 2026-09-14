# Quack client lifetime

Configured-board Quack connections still authenticate and close independently.
Their exact sealed `httpfs` and `quack` images are retained in a process-local
cache because the native loader keeps extension mappings after connection
close. Creating a new memfd image for every connection otherwise retains another
copy of the libraries on every readiness or intent read.

The cache holds at most four extension-set identities without eviction. Each
borrow verifies the caller's regular source projection and the cached immutable
image custody. Every native LOAD retains the before/after checks and filesystem
race guard. A changed or corrupt projection fails closed. Exhausting the image
bound requires the existing qualified process-restart path; repeatedly evicting
and loading images would recreate the memory leak.

Closing a connection releases its image lease. The verified images remain until
process exit. Forked children cannot borrow or delete the parent's images; an
exec creates its own cache. Database attachments, tokens, transactions, owner
generations, and task authority are not cached by this component.
