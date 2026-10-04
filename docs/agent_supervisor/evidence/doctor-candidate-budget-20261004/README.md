# Apply the model timeout reserve only to the model route

The driver used remaining(25) while constructing every implementation command.
Both Doctor candidate commands ignore that timeout, so a completed candidate
could be refused before the actual work cutoff by an unused provider reserve.
The driver now checks remaining() for those two validated candidate routes;
the model route still uses remaining(25). The candidate adapter still checks
status/provider-count bindings. Native admission, publication and validation
remain required; this change grants no repair or completion authority.

Two before-patch tests reproduce the late-candidate refusal. After the patch,
all 110 supervisor tests execute and pass. The new controls run the actual
driver and command adapter with controlled clocks/native-entry seams: candidates
can reach the native boundary with five seconds remaining, model calls retain
their reserve, and every route refuses at the unchanged 245-second work cutoff.
The 40-second cleanup reserve and alarm/cleanup behavior remain intact.

These are component controls, not proof that a native task can finish in the
remaining time. The preceding full trial spent about 240 seconds preparing
indexes, symbolic planning, context and Doctor's checked candidate. Its reward
remains zero. The new driver still needs Docker qualification, and preparation
overhead must fall enough to leave useful dispatch/execution time. No deadline,
resource envelope or proof/source-currentness check has been relaxed.
