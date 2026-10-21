// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! `rmlx tune decide` — honest winner selection over a measured speculative
//! sweep, distilled from MTPLX's `_best_multiplier_summary`
//! (`mtplx/commands/public.py:5587`, upstream `youssofal/MTPLX`).
//!
//! # What this is and is not
//!
//! This is **not** a new benchmarking engine. It is a decision rule applied to
//! rows a caller already measured with `rmlx bench` / `scripts/spec_bench.sh`
//! — one AR (no-drafter) baseline row and one or more candidate rows, each
//! carrying the same [`RoundStats`](rmlx_models::speculative::round_stats)
//! vocabulary already produced by `run_rounds`
//! (`crates/rmlx-models/src/speculative/round_loop.rs:423`) and already
//! recorded by `scripts/lib/spec_round_log.py`. This module reads a small
//! typed summary of those rows from JSON (one object per candidate) and
//! decides, per MTPLX's honesty rules:
//!
//! 1. A candidate is **disqualified outright** if its accept rate collapsed
//!    to ~0 (`TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD` — mirrors MTPLX's
//!    `_tune_row_acceptance_collapsed`, `mtplx/commands/public.py:5729`): with
//!    zero accepted drafts every round does strictly more work than plain
//!    decode for the same output, so any apparent speed-up on such a row can
//!    only be measurement noise, never a true result.
//! 2. Among the rows that are not disqualified, the fastest-by-decode-tps row
//!    must still beat the AR baseline by a real margin (not just equal or
//!    barely above it) before it is declared a winner at all.
//! 3. Within [`TUNE_TIE_PREFER_DEEPER_WITHIN_PCT`] of the fastest surviving
//!    row's tok/s, prefer the **deeper** block size (mirrors MTPLX's
//!    `MTPLX_TUNE_TIE_PREFER_DEEPER_WITHIN_PCT` tie-breaker,
//!    `mtplx/commands/public.py:4378`) — a deeper block has more headroom for
//!    workloads whose future prompts draft further, and ties this close are
//!    noise, not evidence the shallower block is actually faster.
//!
//! If no candidate clears both gates, this reports which gate failed
//! (`TuneVerdict::NoQualifyingCandidate` vs. `TuneVerdict::AllAcceptanceCollapsed`)
//! rather than silently declaring the fastest row of a bad sweep the winner —
//! the same refusal MTPLX's tuner makes when "nothing beat baseline."
//!
//! # Why the sweep itself is out of scope here
//!
//! `rmlx bench --draft-model` does not exist: `BenchArgs`
//! (`crates/rmlx-cli/src/commands/bench.rs:590`) and `Cmd::Bench`
//! (`crates/rmlx-cli/src/main.rs:1384`) carry no draft-model flags at all —
//! only `rmlx serve` and `scripts/spec_bench.sh` know how to run a
//! speculative arm. Wiring an automatic depth-sweep-and-measure loop would
//! mean either adding `--draft-model`/`--draft-block-size` to `rmlx bench`
//! (a server-shaped feature the bench binary was deliberately kept out of) or
//! shelling out to `scripts/spec_bench.sh` per candidate (a second
//! measurement path beside the one this repo's hard rules already hold to a
//! single source of truth, `rmlx metrics record`). Both are real work with
//! real design tradeoffs this module does not make unilaterally. What *is*
//! safe to build without that decision is the part MTPLX's own tuner treats
//! as separable: given rows already measured and already filed, decide the
//! winner honestly. That is what `rmlx tune decide` does.
//!
//! No standalone MTP drafter snapshot exists on this host to validate an
//! end-to-end sweep against (confirmed live: `--draft-model` pointed at the
//! coder checkpoint's own directory is refused by
//! `classify_mtp_draft`/`decide_draft_kind`
//! (`crates/rmlx-server/src/engine/speculative.rs:163`) because the
//! checkpoint's `config.json` declares `Qwen3_5ForConditionalGeneration`, a
//! full generative model, not `qwen3_5_mtp`; the embedded `language_model.mtp.*`
//! tensors have no loader path that splits them out of a full checkpoint into
//! a standalone sidecar directory — `rmlx convert` does not exist yet,
//! `CLAUDE.md:33`). This is a second, independent reason the sweep-and-measure
//! half is out of scope for this change: there is no model on this machine to
//! validate it against, and a path that only compiles is not evidence.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Below this accept rate, a round draws strictly more compute than plain
/// decode for the tokens it emits. Mirrors MTPLX's
/// `TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD` (`mtplx/commands/public.py`, read
/// beside `TUNE_TIE_PREFER_DEEPER_WITHIN_PCT` at line 259) — MTPLX does not
/// publish its exact numeric value in the source excerpt reviewed, so this
/// picks the same order of magnitude their `_tune_row_acceptance_collapsed`
/// docstring describes ("acceptance ~zero"): any rate at or below 2% is
/// treated as collapsed. This is deliberately conservative — a real working
/// drafter accepts well above this on any checkpoint worth serving, and a
/// looser threshold would let through a row whose "beat baseline" is sampling
/// noise on a drafter that is not actually doing useful work.
pub(crate) const TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD: f64 = 0.02;

/// Within this percent of the fastest surviving candidate's tok/s, the
/// decision prefers the row with the larger block size. Mirrors MTPLX's
/// `MTPLX_TUNE_TIE_PREFER_DEEPER_WITHIN_PCT` default
/// (`mtplx/commands/public.py:259`).
pub(crate) const TUNE_TIE_PREFER_DEEPER_WITHIN_PCT: f64 = 2.0;

/// A measured speculative-decode candidate row, built from the same fields
/// `scripts/lib/spec_round_log.py` already reads back from one request's
/// `done` event (`rounds`, `total_draft`, `total_accept`, `block_size`,
/// `decode_tps`) — see `docs/SPECULATIVE.md` §"Benching an arm" table
/// (lines 491-497). This struct does not invent a new measurement vocabulary;
/// it names the subset of an already-filed row this decision rule needs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct TuneCandidate {
    /// The draft block size this row ran at (`RoundStats::block_size`,
    /// `crates/rmlx-models/src/speculative/round_stats.rs:488` — the block the
    /// engine actually resolved and ran, which a sidecar can cap below the one
    /// requested).
    pub(crate) block_size: usize,
    /// Decode tokens/sec over the engine's own measured window (first emitted
    /// token to last, prefill excluded) — the same `decode_tps_warm` metric
    /// `scripts/spec_bench.sh`'s `emit_and_ingest` files
    /// (`scripts/spec_bench.sh:502`).
    pub(crate) decode_tps: f64,
    /// `total_accept / total_draft` for this row
    /// (`RoundStats::accept_rate`, `crates/rmlx-models/src/speculative/round_stats.rs:550`).
    pub(crate) accept_rate: f64,
}

/// The no-drafter baseline row: decode tok/s with no round loop at all.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct TuneBaseline {
    /// Decode tokens/sec, measured the same way as
    /// [`TuneCandidate::decode_tps`] but on the plain (non-speculative) arm —
    /// the server's own derived rate from inter-token gaps
    /// (`scripts/spec_bench.sh` §"Measurement basis" comment).
    pub(crate) decode_tps: f64,
}

/// Why `decide` did or did not name a winner. Every variant is one MTPLX
/// honesty rule this module ports; see the module doc for the rule each one
/// enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TuneVerdict {
    /// A winner was found: at least one candidate was not acceptance-collapsed
    /// and beat the baseline by a real margin.
    Won,
    /// Every candidate's accept rate collapsed to ~0 — none of them is
    /// reporting a result this rule will stand behind, whatever their
    /// apparent tok/s says.
    AllAcceptanceCollapsed,
    /// At least one candidate was not acceptance-collapsed, but none beat the
    /// baseline by more than measurement noise.
    NoQualifyingCandidate,
    /// The candidate list was empty — nothing to decide between.
    NoCandidates,
}

/// The decision `decide` returns: the verdict, and the winning row's
/// `block_size` + `decode_tps` + the margin it won by, when there is one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct TuneDecision {
    pub(crate) verdict: TuneVerdict,
    /// `Some` only when `verdict == Won`.
    pub(crate) winner_block_size: Option<usize>,
    /// `Some` only when `verdict == Won`.
    pub(crate) winner_decode_tps: Option<f64>,
    /// `(winner.decode_tps / baseline.decode_tps - 1.0) * 100.0`. `Some` only
    /// when `verdict == Won`.
    pub(crate) multiplier_pct: Option<f64>,
    /// `true` when the tie-breaker (prefer-deeper) changed which row won,
    /// relative to the single fastest surviving row. Mirrors MTPLX's
    /// `tie_breaker.applied` field (`mtplx/commands/public.py:5679`) — exposed
    /// so a caller can tell "the fastest row won" from "a close second won
    /// because it drafts deeper," which is a different claim to log.
    pub(crate) tie_breaker_applied: bool,
}

impl TuneDecision {
    const fn no_candidates() -> Self {
        Self {
            verdict: TuneVerdict::NoCandidates,
            winner_block_size: None,
            winner_decode_tps: None,
            multiplier_pct: None,
            tie_breaker_applied: false,
        }
    }

    const fn all_collapsed() -> Self {
        Self {
            verdict: TuneVerdict::AllAcceptanceCollapsed,
            winner_block_size: None,
            winner_decode_tps: None,
            multiplier_pct: None,
            tie_breaker_applied: false,
        }
    }

    const fn no_qualifying() -> Self {
        Self {
            verdict: TuneVerdict::NoQualifyingCandidate,
            winner_block_size: None,
            winner_decode_tps: None,
            multiplier_pct: None,
            tie_breaker_applied: false,
        }
    }
}

/// Decide a winner among `candidates` against `baseline`, per the honesty
/// rules in the module doc.
///
/// `min_margin_pct` is the smallest `(candidate.decode_tps / baseline.decode_tps - 1.0) * 100.0`
/// this call will accept as "beat baseline" rather than noise. The caller
/// supplies it rather than this module hard-coding one, because what counts
/// as noise is a property of the measurement instrument (`rmlx bench`'s own
/// `TREND_MAX_DRIFT_PCT` / `SETTLED_MAX_RANGE_PCT`,
/// `crates/rmlx-cli/src/commands/bench.rs:79,135`), not of this decision
/// rule. A caller feeding in rows from `rmlx bench`'s settled-cell gate
/// should pass that instrument's own drift bound here, not invent a second
/// number.
///
/// Ties within [`TUNE_TIE_PREFER_DEEPER_WITHIN_PCT`] of the fastest surviving
/// candidate prefer the larger `block_size`; among equal block sizes, the one
/// named earlier in `candidates` wins (stable, so repeated calls with the same
/// input in the same order are deterministic).
#[must_use]
pub(crate) fn decide(
    baseline: TuneBaseline,
    candidates: &[TuneCandidate],
    min_margin_pct: f64,
) -> TuneDecision {
    if candidates.is_empty() {
        return TuneDecision::no_candidates();
    }

    let survivors: Vec<&TuneCandidate> = candidates
        .iter()
        .filter(|c| c.accept_rate > TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD)
        .collect();

    if survivors.is_empty() {
        return TuneDecision::all_collapsed();
    }

    // Guard a non-positive or non-finite baseline before it reaches a
    // division: a zero/negative/NaN baseline would make every multiplier
    // either undefined or spuriously "infinite," which is not a real speed-up
    // — it is an unmeasurable baseline the caller must fix upstream.
    if !(baseline.decode_tps.is_finite() && baseline.decode_tps > 0.0) {
        return TuneDecision::no_qualifying();
    }

    let qualifying: Vec<&TuneCandidate> = survivors
        .iter()
        .copied()
        .filter(|c| {
            c.decode_tps.is_finite()
                && margin_pct(baseline.decode_tps, c.decode_tps) > min_margin_pct
        })
        .collect();

    if qualifying.is_empty() {
        return TuneDecision::no_qualifying();
    }

    // Fastest-by-tok/s among qualifying rows. `max_by` keeps the first
    // maximal element on a tie (stable), matching the "earlier in the slice
    // wins equal block sizes" contract documented above.
    //
    // `qualifying` was just checked non-empty above, so `max_by` always
    // returns `Some` here; the `let...else` still names a real (if
    // unreachable) exit instead of `.expect()`, which this crate's lints
    // forbid even where a panic genuinely cannot happen.
    let Some(raw_winner) = qualifying
        .iter()
        .copied()
        .max_by(|a, b| a.decode_tps.total_cmp(&b.decode_tps))
    else {
        return TuneDecision::no_qualifying();
    };

    let tie_floor = raw_winner.decode_tps * (1.0 - TUNE_TIE_PREFER_DEEPER_WITHIN_PCT / 100.0);
    let tied: Vec<&TuneCandidate> = qualifying
        .iter()
        .copied()
        .filter(|c| c.decode_tps >= tie_floor)
        .collect();

    // Prefer the deepest block_size among the tied set; ties on block_size
    // keep the first (stable `max_by`), matching raw_winner's own tie rule.
    // `tied` always contains at least `raw_winner` itself (it satisfies its
    // own floor), so this `max_by` also always returns `Some`.
    let Some(winner) = tied.iter().copied().max_by(|a, b| {
        a.block_size
            .cmp(&b.block_size)
            .then(a.decode_tps.total_cmp(&b.decode_tps))
    }) else {
        return TuneDecision::no_qualifying();
    };

    let tie_breaker_applied = winner.block_size != raw_winner.block_size;

    TuneDecision {
        verdict: TuneVerdict::Won,
        winner_block_size: Some(winner.block_size),
        winner_decode_tps: Some(winner.decode_tps),
        multiplier_pct: Some(margin_pct(baseline.decode_tps, winner.decode_tps)),
        tie_breaker_applied,
    }
}

/// `(candidate / baseline - 1.0) * 100.0`.
fn margin_pct(baseline_tps: f64, candidate_tps: f64) -> f64 {
    (candidate_tps / baseline_tps - 1.0) * 100.0
}

/// One sweep's input: a baseline and a map of label -> candidate, read from a
/// JSON file by `rmlx tune decide --file <path>`. The map (rather than a
/// `Vec`) is the on-disk shape because a caller writing this by hand (or from
/// `scripts/spec_bench.sh` output) names each row by its own cell label
/// (`decode_config`, per `docs/SPECULATIVE.md:494`) — losing that label would
/// make a later "which row won" answer unreadable without re-deriving it from
/// `block_size` alone, which is not always unique across drafter kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TuneSweepInput {
    pub(crate) baseline: TuneBaseline,
    pub(crate) candidates: BTreeMap<String, TuneCandidate>,
}

// ── CLI entry point ─────────────────────────────────────────────────────────

/// `rmlx tune decide --file <path> [--min-margin-pct N] [--json]`.
///
/// Reads a [`TuneSweepInput`] JSON file, applies [`decide`], and prints the
/// result. Always exits 0 for a well-formed input — a `TuneVerdict` other than
/// `Won` is a correct answer ("this sweep found no real win"), not a command
/// failure. Malformed JSON or an unreadable file is the only error path.
pub(crate) fn run_tune_decide(
    file: &std::path::Path,
    min_margin_pct: f64,
    json: bool,
) -> anyhow::Result<i32> {
    let raw = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", file.display()))?;
    let input: TuneSweepInput = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("parsing {} as a tune sweep input: {e}", file.display()))?;

    // Candidates come out of a BTreeMap, so this order is deterministic
    // (sorted by label) rather than JSON-object-insertion-order, which serde
    // does not guarantee is preserved anyway.
    let rows: Vec<TuneCandidate> = input.candidates.values().copied().collect();
    let n_rows = rows.len();

    let decision = decide(input.baseline, &rows, min_margin_pct);

    if json {
        let rendered = serde_json::to_string(&decision)
            .map_err(|e| anyhow::anyhow!("serialising decision: {e}"))?;
        println!("{rendered}");
        return Ok(0);
    }

    println!("baseline decode_tps: {:.3}", input.baseline.decode_tps);
    println!("candidates measured: {n_rows}");
    println!("min_margin_pct:      {min_margin_pct:.2}");
    match decision.verdict {
        TuneVerdict::Won => {
            // `decide` only reaches `Won` with every `winner_*` field `Some`
            // (see `TuneDecision`'s doc comments); a well-formed value here
            // that is missing one is this module's own bug, not a condition
            // this CLI path can recover from, so the match falls through to
            // the generic "no winner" line instead of panicking.
            if let (Some(block), Some(tps), Some(mult)) = (
                decision.winner_block_size,
                decision.winner_decode_tps,
                decision.multiplier_pct,
            ) {
                // Find the label this block_size+decode_tps came from, for a
                // readable summary line (best-effort: if two rows share both
                // values, the first match in sorted-label order is named).
                let label = input
                    .candidates
                    .iter()
                    .find(|(_, c)| {
                        c.block_size == block && (c.decode_tps - tps).abs() < f64::EPSILON
                    })
                    .map_or("(unlabeled)", |(k, _)| k.as_str());
                println!("verdict: WON");
                println!("  winner:        {label} (block_size={block})");
                println!("  decode_tps:    {tps:.3}");
                println!("  margin:        +{mult:.2}% vs baseline");
                println!(
                    "  tie_breaker:   {}",
                    if decision.tie_breaker_applied {
                        "applied (deeper block preferred within tie window)"
                    } else {
                        "not applied"
                    }
                );
            } else {
                println!(
                    "verdict: WON (internal inconsistency: winner fields missing — this is a bug)"
                );
            }
        }
        TuneVerdict::AllAcceptanceCollapsed => {
            println!("verdict: ALL_ACCEPTANCE_COLLAPSED");
            println!(
                "  every candidate's accept rate was at or below {:.1}% — \
                 none of them is a real result, whatever their tok/s says.",
                TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD * 100.0
            );
        }
        TuneVerdict::NoQualifyingCandidate => {
            println!("verdict: NO_QUALIFYING_CANDIDATE");
            println!(
                "  at least one candidate had a healthy accept rate, but none beat \
                 baseline by more than {min_margin_pct:.2}% — not a real win."
            );
        }
        TuneVerdict::NoCandidates => {
            println!("verdict: NO_CANDIDATES");
            println!("  the input file named no candidate rows to decide between.");
        }
    }

    Ok(0)
}

#[cfg(test)]
#[path = "tune_tests.rs"]
mod tune_tests;
