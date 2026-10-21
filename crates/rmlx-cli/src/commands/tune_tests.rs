//! Unit tests for the `rmlx tune decide` decision rule
//! (`crates/rmlx-cli/src/commands/tune.rs`).
//!
//! No GPU, no model, no server — `decide` is a pure function over typed
//! numbers, so every case here is a synthetic row. The only way these tests
//! could pass without the rule being correct is if the arithmetic itself is
//! wrong, which is exactly what they check.

use super::{decide, TuneBaseline, TuneCandidate, TuneDecision, TuneVerdict};

fn baseline(tps: f64) -> TuneBaseline {
    TuneBaseline { decode_tps: tps }
}

fn candidate(block_size: usize, decode_tps: f64, accept_rate: f64) -> TuneCandidate {
    TuneCandidate {
        block_size,
        decode_tps,
        accept_rate,
    }
}

#[test]
fn empty_candidates_reports_no_candidates() {
    let d = decide(baseline(50.0), &[], 5.0);
    assert_eq!(d.verdict, TuneVerdict::NoCandidates);
    assert_eq!(d.winner_block_size, None);
    assert_eq!(d.winner_decode_tps, None);
    assert_eq!(d.multiplier_pct, None);
    assert!(!d.tie_breaker_applied);
}

#[test]
fn every_candidate_acceptance_collapsed_reports_that_verdict_even_if_fast() {
    // Both rows report a huge tok/s — if acceptance gating were skipped, the
    // 200.0 row would look like a 4x win. It must not: zero accepted drafts
    // means the round did strictly more work than plain decode, so this can
    // only be noise.
    let rows = [candidate(4, 200.0, 0.0), candidate(8, 150.0, 0.01)];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::AllAcceptanceCollapsed);
    assert_eq!(d.winner_block_size, None);
}

#[test]
fn accept_rate_exactly_at_threshold_is_still_collapsed() {
    // The gate is a strict `>`, so a row sitting exactly on the threshold is
    // not a survivor — this pins that boundary rather than leaving it to
    // float-comparison luck.
    let rows = [candidate(
        4,
        200.0,
        super::TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD,
    )];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::AllAcceptanceCollapsed);
}

#[test]
fn accept_rate_just_above_threshold_survives_the_collapse_gate() {
    let rows = [candidate(
        4,
        100.0,
        super::TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD + 0.001,
    )];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
}

#[test]
fn surviving_candidate_that_never_beats_baseline_margin_reports_no_qualifying() {
    // Accept rate is healthy, but tok/s barely moves — below the caller's own
    // margin of "not noise."
    let rows = [
        candidate(4, 51.0, 0.6), // +2% vs baseline, under the 5% floor
        candidate(8, 50.5, 0.55),
    ];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::NoQualifyingCandidate);
    assert_eq!(d.winner_block_size, None);
}

#[test]
fn margin_exactly_at_floor_does_not_qualify_strict_gate() {
    // Compute the candidate's margin exactly as `decide` itself would, then
    // use that same float as `min_margin_pct` — this pins the boundary is a
    // strict `>` without depending on 55.0/50.0 happening to round to
    // exactly 10.0 in IEEE 754 (it does not: it rounds to
    // 10.000000000000009, so a hand-picked "10.0" would pass this gate for
    // the wrong reason).
    let base = 50.0;
    let cand_tps = 55.0;
    let exact_margin = (cand_tps / base - 1.0) * 100.0;
    let rows = [candidate(4, cand_tps, 0.6)];
    let d = decide(baseline(base), &rows, exact_margin);
    assert_eq!(d.verdict, TuneVerdict::NoQualifyingCandidate);
}

#[test]
fn margin_just_above_floor_qualifies() {
    let rows = [candidate(4, 55.01, 0.6)];
    let d = decide(baseline(50.0), &rows, 10.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
    assert_eq!(d.winner_block_size, Some(4));
}

#[test]
fn clear_winner_reports_correct_block_and_multiplier() {
    let rows = [
        candidate(2, 70.0, 0.5),
        candidate(4, 90.0, 0.6), // fastest, clear winner
        candidate(6, 80.0, 0.55),
    ];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
    assert_eq!(d.winner_block_size, Some(4));
    assert_eq!(d.winner_decode_tps, Some(90.0));
    // (90 / 50 - 1) * 100 = 80.0
    let mult = d.multiplier_pct.expect("winner present");
    assert!((mult - 80.0).abs() < 1e-9, "multiplier_pct = {mult}");
    assert!(!d.tie_breaker_applied);
}

#[test]
fn acceptance_collapsed_row_is_excluded_even_when_it_is_the_fastest() {
    // The 4-block row is fastest in raw tok/s but its accept rate collapsed —
    // it must be excluded entirely, leaving the slower-but-healthy 6-block row
    // as the winner. If the collapse gate were applied after picking the
    // fastest row instead of before, this would wrongly report block_size 4.
    let rows = [
        candidate(4, 999.0, 0.0), // collapsed — must not win
        candidate(6, 80.0, 0.55), // only real survivor
    ];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
    assert_eq!(d.winner_block_size, Some(6));
    assert_eq!(d.winner_decode_tps, Some(80.0));
}

#[test]
fn tie_breaker_prefers_deeper_block_within_pct_window() {
    // Fastest raw winner is block_size=3 at 100.0 tok/s. block_size=6 is at
    // 99.0 tok/s, which is within TUNE_TIE_PREFER_DEEPER_WITHIN_PCT (2%) of
    // 100.0 (floor = 98.0), so the deeper block should win instead.
    let rows = [candidate(3, 100.0, 0.6), candidate(6, 99.0, 0.55)];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
    assert_eq!(d.winner_block_size, Some(6));
    assert!(d.tie_breaker_applied);
}

#[test]
fn tie_breaker_does_not_apply_when_margin_is_outside_the_window() {
    // block_size=6 is well outside the 2% tie window around the fastest row
    // (100.0 * 0.98 = 98.0 floor; 90.0 is below it), so the fastest row wins
    // outright and the tie-breaker never engages.
    let rows = [candidate(3, 100.0, 0.6), candidate(6, 90.0, 0.55)];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
    assert_eq!(d.winner_block_size, Some(3));
    assert!(!d.tie_breaker_applied);
}

#[test]
fn tie_breaker_not_applied_when_fastest_row_is_already_deepest() {
    // The deepest row is also the fastest, so even though it is within its
    // own tie window (trivially, against itself), the winner equals the raw
    // winner and tie_breaker_applied must be false.
    let rows = [candidate(3, 90.0, 0.6), candidate(6, 100.0, 0.55)];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.winner_block_size, Some(6));
    assert!(!d.tie_breaker_applied);
}

#[test]
fn equal_decode_tps_keeps_the_earlier_row_stable() {
    // Two rows tie exactly on tok/s with the same block_size recorded twice
    // (a caller re-measuring the same cell) — `max_by` must deterministically
    // keep one of them, not panic or alternate between runs.
    let rows = [candidate(4, 90.0, 0.5), candidate(4, 90.0, 0.6)];
    let d1 = decide(baseline(50.0), &rows, 5.0);
    let d2 = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d1, d2, "decide must be deterministic on identical input");
    assert_eq!(d1.winner_block_size, Some(4));
}

#[test]
fn nonpositive_baseline_reports_no_qualifying_rather_than_dividing() {
    let rows = [candidate(4, 90.0, 0.6)];
    for bad_baseline in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let d = decide(baseline(bad_baseline), &rows, 5.0);
        assert_eq!(
            d.verdict,
            TuneVerdict::NoQualifyingCandidate,
            "baseline={bad_baseline} should refuse rather than divide"
        );
    }
}

#[test]
fn nonfinite_candidate_decode_tps_is_excluded_from_qualifying() {
    let rows = [
        candidate(4, f64::NAN, 0.6),
        candidate(6, f64::INFINITY, 0.6),
        candidate(8, 80.0, 0.6),
    ];
    let d = decide(baseline(50.0), &rows, 5.0);
    assert_eq!(d.verdict, TuneVerdict::Won);
    assert_eq!(d.winner_block_size, Some(8));
}

#[test]
fn single_survivor_below_margin_and_single_collapsed_distinguishes_verdicts() {
    // A collapsed-only sweep and a healthy-but-slow-only sweep must report
    // two DIFFERENT verdicts — a caller distinguishing "the drafter never
    // worked" from "the drafter worked but wasn't worth it" needs these to
    // not collapse into one generic "no" result.
    let collapsed_only = [candidate(4, 200.0, 0.0)];
    let slow_only = [candidate(4, 51.0, 0.6)];
    let d_collapsed = decide(baseline(50.0), &collapsed_only, 5.0);
    let d_slow = decide(baseline(50.0), &slow_only, 5.0);
    assert_eq!(d_collapsed.verdict, TuneVerdict::AllAcceptanceCollapsed);
    assert_eq!(d_slow.verdict, TuneVerdict::NoQualifyingCandidate);
    assert_ne!(d_collapsed.verdict, d_slow.verdict);
}

#[test]
fn tune_sweep_input_round_trips_through_json() {
    use super::TuneSweepInput;
    use std::collections::BTreeMap;

    let mut candidates = BTreeMap::new();
    candidates.insert("mtp/block=4".to_owned(), candidate(4, 90.0, 0.6));
    candidates.insert("mtp/block=6".to_owned(), candidate(6, 80.0, 0.55));
    let input = TuneSweepInput {
        baseline: baseline(50.0),
        candidates,
    };

    let json = serde_json::to_string(&input).expect("serialize");
    let back: TuneSweepInput = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.baseline.decode_tps, input.baseline.decode_tps);
    assert_eq!(back.candidates.len(), input.candidates.len());
    assert_eq!(
        back.candidates["mtp/block=4"].decode_tps,
        input.candidates["mtp/block=4"].decode_tps
    );
}

#[test]
fn decision_serde_round_trips_every_verdict_variant() {
    // Pins that every TuneVerdict variant (including the three "no winner"
    // ones that carry no payload) survives a JSON round trip, since the CLI
    // command surfaces this struct directly as `--json` output.
    let decisions = [
        TuneDecision {
            verdict: TuneVerdict::Won,
            winner_block_size: Some(4),
            winner_decode_tps: Some(90.0),
            multiplier_pct: Some(80.0),
            tie_breaker_applied: false,
        },
        decide(baseline(50.0), &[], 5.0),
        decide(baseline(50.0), &[candidate(4, 200.0, 0.0)], 5.0),
        decide(baseline(50.0), &[candidate(4, 51.0, 0.6)], 5.0),
    ];
    for d in decisions {
        let json = serde_json::to_string(&d).expect("serialize");
        let back: TuneDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, d);
    }
}
