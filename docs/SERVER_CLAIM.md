# HTTP server: Metal claim file

Split out of [`SERVER.md`](SERVER.md) to keep that doc under the size cap.

## Claim file

`claim::try_claim() -> Result<MetalClaim, ClaimError>` guards one
machine-wide Metal claim: an exclusive `flock` on `/var/tmp/rmlx.claim`.
`/var/tmp` is machine-wide and outlives macOS's `tmp_cleaner`, which only
sweeps `/tmp`.

1. It opens (or creates, mode `0666`) the file with `O_NOFOLLOW | O_CLOEXEC |
   O_NONBLOCK`, and refuses anything that is not a regular file with one
   link. A write open on another user's file gets `EACCES`, and it falls
   back to a read-only open.
2. It takes an exclusive, non-blocking `flock` on it — a read-only open
   takes the same `flock` as a writable one.
3. If the open was writable, it truncates the file and writes `<pid>
   <argv>` into it. A read-only claimer skips this step, so its body still
   names whichever process last held it open for writing.

If another process already holds the `flock`, `try_claim` returns
`ClaimError::AlreadyHeld { holder_pid, holder_command, path }`, read from the
body the holder wrote. The `flock` is the only gate — there is no PID
liveness probe. The kernel releases a dead holder's lock the moment it exits
or is killed; the next claimer locks the same file with no reclaim logic and
no operator action.

`MetalClaim` is a RAII guard. Dropping it releases the `flock`, unless a
child it passed the fd to (`rmlx claim run`) still runs — that child's own
copy of the fd keeps the lock held. The file itself is never deleted. `rmlx
serve` drops the guard at the end of its shutdown, unless a job still runs:
then the lock stays until the process is gone. `rmlx claim run -- <command>`
holds the claim for `<command>`'s whole life; see `docs/CLI.md`
§ "Claim file".

**The claim is machine-wide, not per port or per command.** Every GPU
command — `serve`, `chat`, `baseline`, `bench`, `kv-calibrate`,
`healthcheck --full`, and so on — takes the same claim, so only one can hold
the GPU at a time; a second is refused with exit code `11`, naming the
recorded holder. The lock is advisory and does not see non-rMLX MLX
processes such as `mlx_lm.server`; the `ClaimError` message prints a stop
hint (`kill <PID>`), never a delete hint.

**Legacy per-port claim (one release).** Older builds held
`/tmp/rmlx.<port>.claim`. Before it takes the machine-wide claim, this build
opens each such file and takes a brief shared lock on it, released at once —
enough to tell whether it is held, not to hold it. It skips a vanished
entry, a symlink, or anything that is not a regular file. Any other open
error there refuses every GPU command with an I/O error, not exit `11`.
Nothing is deleted. Removed in the release after next.

## Experimental multi-process memory budget

`rmlx serve --experimental-multi-process-budget <GB_f64>` opts one server
into a separate machine-wide reservation ledger at
`/var/tmp/rmlx.budget.json`. With the flag absent, `serve` still calls
`try_claim()` and retains the exclusive `/var/tmp/rmlx.claim` flock and its
exit-code-11 refusal behavior. There is no environment-variable switch. Use
`--device gpu`; the flag is refused for CPU mode.

The ledger's `available_gb` is physical RAM reported by macOS `hw.memsize`
minus the fixed 15% OS headroom. Each entry records a unique id, PID, requested
GiB, and argv. Registration and release read-modify-write the same regular,
single-link ledger file while holding its exclusive `flock`; the file is never
removed. A reservation that does not fit returns a named budget refusal and
CLI exit code `12`. Each live process removes its entry on normal Drop. A later
Both the default exclusive path and the budget path arbitrate through the
existing `/var/tmp/rmlx.claim` flock: default mode holds it for its lifetime
after confirming the ledger is empty; budget mode holds it through the
ledger check-and-register transaction. This prevents mixed-mode overlap and
serializes a switch between modes. A probe also checks the ledger while
holding the coordination flock.

The ledger registrant checks PIDs with `kill(pid, 0)` and reaps entries whose process is
gone; an indeterminate liveness check is retained conservatively. If a server
shutdown deadline leaves a blocking GPU job running, its reservation remains
until a later ledger operation observes that the process has exited.

This is an **experimental reservation**, not a measured peak-memory guard.
It counts the operator-supplied per-process amount; it does not measure Metal
peak allocation, `phys_footprint`, model-load transients, or prompt-cache use.
Do not run multiple real model processes until the G2.6 memory preflight is
complete. To roll back, stop all budget-mode processes before launching
without the flag; the default exclusive mode cannot coordinate with a budget
process that is still live. This code does not remove or rewrite the legacy
claim file and is not a production go-live approval.
