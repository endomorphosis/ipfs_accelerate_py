# Worldwide legal search recovery

Keep the registered execute-mode watchdog running during maintenance. Use a
bounded `rescue.maintenance_hold.maintenance_hold` context instead of replacing
the watchdog command with an observation-only command. Manual and unrecognized
holds require explicit release. Native pause snapshots bind leased maintenance
markers to their exact release archives, so an expired dead owner's pause can
clear without discarding an unrelated native hold.

`continue-candidate` admits one exact committed candidate through the existing
Native claim, independent validators, merge train and completion CAS. It does
not regenerate the candidate, refund its failed predecessor or replenish any
attempt budget. Its owned, hash-bound pending request survives a service restart.
The continuous service removes that pointer only after a real native completion
receipt and then reopens the ordinary task frontier.

Git observations have a 60-second bound; checkout creation has a 180-second
bound. A checkout interruption requests a registered service restart so its
whole cgroup drains. On restart, only the exact request-derived PREPARING
lifecycle can use Native's dead-owner CAS. The partial checkout is retained.
Native startup may already have fenced that same dead owner; its exact terminal
record is also accepted only while the verified event journal proves the
candidate never crossed the checkout-prepared execution boundary.
There is one separate checkout retry, with a distinct request-bound path and
the original task, claim, validation and completion authority. An unregistered
directory, live owner, different lifecycle, active validation, changed candidate
or exhausted checkout retry refuses this recovery.

An unknown ordinary provider callback remains unknown and is never invoked
again by this recovery. Native can reconcile an existing completed Portal
projection using its genuine completion event and accepted-source evidence.
The sole execution replay is the explicit no-provider candidate checkout above.
The original callback intent remains in the reconciliation event; acceptance
still passes through the existing native receipt validation and completion CAS.

The one-shot command returns failure when native completion was not verified.
The continuous service uses exit 75 for a fenced checkout restart or to reopen
the unsliced frontier after candidate completion. Heartbeats and a successful
process exit do not count as completed work.
