// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! `rmlx info --estimate-memory` — byte-exact preflight memory estimate for a
//! model snapshot, no MLX runtime / Metal context required.
//!
//! Prints `weights_bytes` (read from the shard index's declared `total_size`,
//! or summed from on-disk shard file sizes when the snapshot has no index),
//! `kv_bytes_total` (derived from [`rmlx_kv_quant::KvQuant::estimated_resident_bytes_per_layer`]
//! at the requested `--max-tokens`, counting only full-attention layers for a
//! hybrid snapshot), a fixed `activation_bytes` allowance, `headroom_bytes`
//! (15% of physical RAM, matching the Gate 2 OS-headroom convention already
//! used by the multi-process budget ledger), `budget_bytes` (physical RAM
//! minus headroom), `total_bytes` (weights + KV + activation) and `fits`
//! (whether `total_bytes <= budget_bytes`).
//!
//! This does not model Metal driver transient allocation, prompt-cache growth
//! beyond one request, or prefill chunk buffers precisely — it is a
//! conservative preflight estimate, not a guarantee. Read `docs/CLI.md`
//! § "Memory estimate" for the exact formula and its known gaps.
//!
//! # Public API
//!
//! - [`run_estimate_memory`] — main entry point; prints the estimate and
//!   returns the process exit code (`0` = fits, `1` = does not fit, `3` =
//!   could not resolve the snapshot's architecture facts needed to estimate).

use std::io::Write;
use std::path::Path;

use rmlx_core::system::physical_memory_bytes;
use rmlx_kv_quant::KvQuant;
use rmlx_loader::{load_config, load_shard_index, ModelConfig};

/// Fraction of physical RAM reserved as OS headroom, matching the 15%
/// convention already chosen for the multi-process budget ledger (Gate 2 of
/// `rust-inference-landscape.html`'s multi-agent plan).
const HEADROOM_FRACTION: f64 = 0.15;

/// Fixed per-request activation/transient allowance in bytes (768 MiB).
///
/// This is a conservative constant, not a per-arch measurement: prefill
/// scratch buffers, the MLX graph's intermediate tensors and the Metal
/// command-buffer overhead all vary by architecture and context length in
/// ways this estimator does not model byte-for-byte. Gate 2 of the
/// multi-agent plan measured Metal peak allocation at 18.2 GB against a
/// weights size of ~17.0 GB for the Qwen3.8-27B coder checkpoint at `auto`
/// KV-quant and a 16384-token ceiling — roughly 1.2 GB of headroom over
/// weights alone at that context size, which this constant stays under by
/// design (fails conservatively toward "does not fit" rather than optimistically).
const ACTIVATION_BYTES_ALLOWANCE: u64 = 768 * 1024 * 1024;

/// One line of the `--estimate-memory` report.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryEstimate {
    /// Total on-disk weight bytes, from the shard index's `metadata.total_size`
    /// when present, else the sum of shard file sizes.
    pub(crate) weights_bytes: u64,
    /// Estimated resident KV-cache bytes at `max_tokens`, full-attention
    /// layers only (a hybrid snapshot's linear/SSM layers hold no KV).
    pub(crate) kv_bytes_total: u64,
    /// Fixed allowance for prefill/decode transient activations.
    pub(crate) activation_bytes: u64,
    /// `HEADROOM_FRACTION` of physical RAM, reserved for the OS and not
    /// counted as available budget.
    pub(crate) headroom_bytes: u64,
    /// Physical RAM minus `headroom_bytes` — the ceiling this estimate is
    /// checked against.
    pub(crate) budget_bytes: u64,
    /// `weights_bytes + kv_bytes_total + activation_bytes`.
    pub(crate) total_bytes: u64,
    /// `total_bytes <= budget_bytes`.
    pub(crate) fits: bool,
    /// Number of layers counted as holding KV (full-attention layers for a
    /// hybrid snapshot; `num_hidden_layers` for a uniform-attention one).
    pub(crate) kv_layers_counted: u64,
    /// Total decoder layers declared by `config.json`, for context in the
    /// printed report (may exceed `kv_layers_counted` on a hybrid snapshot).
    pub(crate) total_layers: u64,
}

/// Sum of on-disk shard file sizes, used when the snapshot has no
/// `model.safetensors.index.json` `metadata.total_size` field (a single-file
/// snapshot's synthetic index carries no such metadata).
fn sum_shard_file_sizes(model_path: &Path, idx: &rmlx_loader::ShardIndex) -> std::io::Result<u64> {
    let mut seen = std::collections::BTreeSet::new();
    let mut total = 0_u64;
    for shard_name in idx.weight_map.values() {
        if seen.insert(shard_name.clone()) {
            let meta = std::fs::metadata(model_path.join(shard_name))?;
            total = total.saturating_add(meta.len());
        }
    }
    Ok(total)
}

/// Read `weights_bytes`: prefer the shard index's declared `total_size`
/// (present on every multi-shard HuggingFace snapshot's
/// `model.safetensors.index.json`), falling back to summed on-disk shard
/// sizes when that metadata field is absent (single-shard snapshots carry a
/// synthetic index with no `metadata` block).
fn resolve_weights_bytes(model_path: &Path, idx: &rmlx_loader::ShardIndex) -> anyhow::Result<u64> {
    if let Some(total) = idx
        .metadata
        .get("total_size")
        .and_then(serde_json::Value::as_u64)
    {
        return Ok(total);
    }
    sum_shard_file_sizes(model_path, idx)
        .map_err(|e| anyhow::anyhow!("cannot read shard file sizes: {e}"))
}

/// Count of layers whose attention holds a real KV cache.
///
/// `layer_types` (when present — Qwen3.5-family hybrids, Gemma-family SWA
/// configs) names each layer `"full_attention"` or a non-attending kind
/// (`"linear_attention"`, `"sliding_attention"` counted separately by the
/// caller if ever needed); only `"full_attention"` entries hold the
/// `seq_len`-scaling KV this estimator sizes. Without `layer_types`, every
/// decoder layer is assumed to hold KV (the common case: Qwen2/Qwen3 dense,
/// Gemma3, BitNet, Laguna).
fn count_kv_layers(cfg: &ModelConfig) -> (u64, u64) {
    let text = cfg.text_config.as_ref();
    let total_layers = text
        .and_then(|tc| tc.num_hidden_layers)
        .map_or(0, u64::from);

    let kv_layers = text
        .and_then(|tc| tc.layer_types.as_ref())
        .map_or(total_layers, |types| {
            types
                .iter()
                .filter(|t| t.as_str() == "full_attention")
                .count() as u64
        });

    (kv_layers, total_layers)
}

/// Compute the full [`MemoryEstimate`] for `model_path` at `max_tokens`
/// context, under `kv_quant`.
///
/// Returns `Err` when `config.json`'s `text_config` lacks the fields this
/// estimator needs (`num_hidden_layers`, `num_attention_heads`, `head_dim`) —
/// this happens for a config this build cannot yet resolve head_dim or layer
/// count for, and the caller should treat that as "cannot estimate", not as
/// "fits" or "does not fit".
pub(crate) fn estimate(
    model_path: &Path,
    max_tokens: u64,
    kv_quant: KvQuant,
) -> anyhow::Result<MemoryEstimate> {
    let cfg = load_config(model_path).map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
    let idx = load_shard_index(model_path).map_err(|e| anyhow::anyhow!("load_shard_index: {e}"))?;

    let weights_bytes = resolve_weights_bytes(model_path, &idx)?;

    let head_dim = cfg
        .head_dim()
        .ok_or_else(|| anyhow::anyhow!("config.json has no resolvable head_dim — cannot size KV"))?
        as u64;
    let kv_heads = u64::from(
        cfg.text_config
            .as_ref()
            .and_then(|tc| tc.num_key_value_heads)
            .or_else(|| {
                cfg.text_config
                    .as_ref()
                    .and_then(|tc| tc.num_attention_heads)
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "config.json has no num_key_value_heads / num_attention_heads — cannot size KV"
                )
            })?,
    );

    let (kv_layers_counted, total_layers) = count_kv_layers(&cfg);
    if kv_layers_counted == 0 {
        return Err(anyhow::anyhow!(
            "config.json's text_config has no num_hidden_layers — cannot size KV"
        ));
    }

    // `shares_kv=false`: the conservative (larger) estimate. A snapshot whose
    // KV codec actually shares K/V storage across layers (Mixed/RotK boundary
    // layers) would resident fewer bytes than this reports — the estimator's
    // documented bias is toward "overestimate, not underestimate".
    let per_layer_bytes = kv_quant.estimated_resident_bytes_per_layer(
        max_tokens, head_dim, kv_heads, /* shares_kv */ false,
    );
    let kv_bytes_total = per_layer_bytes.saturating_mul(kv_layers_counted);

    let physical_bytes =
        physical_memory_bytes().map_err(|e| anyhow::anyhow!("cannot read physical RAM: {e}"))?;
    #[allow(
        clippy::cast_precision_loss,
        reason = "physical RAM fits comfortably in f64's 52-bit mantissa for any real machine"
    )]
    let headroom_bytes = (physical_bytes as f64 * HEADROOM_FRACTION).round() as u64;
    let budget_bytes = physical_bytes.saturating_sub(headroom_bytes);

    let total_bytes = weights_bytes
        .saturating_add(kv_bytes_total)
        .saturating_add(ACTIVATION_BYTES_ALLOWANCE);

    Ok(MemoryEstimate {
        weights_bytes,
        kv_bytes_total,
        activation_bytes: ACTIVATION_BYTES_ALLOWANCE,
        headroom_bytes,
        budget_bytes,
        total_bytes,
        fits: total_bytes <= budget_bytes,
        kv_layers_counted,
        total_layers,
    })
}

/// Print `estimate` as human-readable text to `out`.
fn print_human(
    out: &mut impl Write,
    estimate: &MemoryEstimate,
    max_tokens: u64,
) -> std::io::Result<()> {
    writeln!(out, "memory estimate (--max-tokens {max_tokens}):")?;
    writeln!(
        out,
        "  weights_bytes:     {:>14}  ({:.2} GiB)",
        estimate.weights_bytes,
        gib(estimate.weights_bytes)
    )?;
    writeln!(
        out,
        "  kv_bytes_total:    {:>14}  ({:.2} GiB, {} of {} layers hold KV)",
        estimate.kv_bytes_total,
        gib(estimate.kv_bytes_total),
        estimate.kv_layers_counted,
        estimate.total_layers
    )?;
    writeln!(
        out,
        "  activation_bytes:  {:>14}  ({:.2} GiB, fixed allowance)",
        estimate.activation_bytes,
        gib(estimate.activation_bytes)
    )?;
    writeln!(
        out,
        "  total_bytes:       {:>14}  ({:.2} GiB)",
        estimate.total_bytes,
        gib(estimate.total_bytes)
    )?;
    writeln!(out)?;
    writeln!(
        out,
        "  headroom_bytes:    {:>14}  ({:.2} GiB, 15% of physical RAM)",
        estimate.headroom_bytes,
        gib(estimate.headroom_bytes)
    )?;
    writeln!(
        out,
        "  budget_bytes:      {:>14}  ({:.2} GiB, physical RAM minus headroom)",
        estimate.budget_bytes,
        gib(estimate.budget_bytes)
    )?;
    writeln!(out)?;
    writeln!(out, "  fits: {}", estimate.fits)?;
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "byte counts well under 2^52 for any real model/RAM size"
)]
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// `rmlx info --estimate-memory` entry point.
///
/// Prints the estimate (human-readable, or JSON with `--json`) and returns
/// the process exit code: `0` when `fits`, `1` when it does not, `3` when
/// the snapshot's config lacks a field this estimator needs.
pub(crate) fn run_estimate_memory(
    model_path: &Path,
    max_tokens: u64,
    kv_quant: KvQuant,
    json: bool,
) -> anyhow::Result<i32> {
    let est = match estimate(model_path, max_tokens, kv_quant) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return Ok(3);
        }
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if json {
        writeln!(
            out,
            "{{\"weights_bytes\":{},\"kv_bytes_total\":{},\"activation_bytes\":{},\"headroom_bytes\":{},\"budget_bytes\":{},\"total_bytes\":{},\"fits\":{},\"kv_layers_counted\":{},\"total_layers\":{},\"max_tokens\":{}}}",
            est.weights_bytes,
            est.kv_bytes_total,
            est.activation_bytes,
            est.headroom_bytes,
            est.budget_bytes,
            est.total_bytes,
            est.fits,
            est.kv_layers_counted,
            est.total_layers,
            max_tokens
        )?;
    } else {
        print_human(&mut out, &est, max_tokens)?;
    }

    Ok(i32::from(!est.fits))
}

#[path = "memory_estimate_tests.rs"]
#[cfg(test)]
mod memory_estimate_tests;
