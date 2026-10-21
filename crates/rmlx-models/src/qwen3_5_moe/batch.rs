//! Batched decode primitive for Qwen3.5-family hybrid models.
//!
//! The request scheduler owns one KV/recurrent cache set per active request.
//! This module combines the work that is naturally batch-safe (embedding,
//! layer norms, GatedDeltaNet recurrence and MLP/MoE projections) while
//! deliberately keeping full-attention cache updates row-wise. Row-wise
//! attention preserves each request's independent offset and keeps all
//! single-request KV codecs (including Rotor/Iso/Planar paths) on their
//! existing B=1 implementation.
//!
//! This is the model-side kernel boundary. The HTTP scheduler integration is a
//! separate layer: no server flag opts into this path until that integration
//! and its end-to-end queue/stream tests are present.

#![allow(
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    clippy::expect_used,
    reason = "batch and layer dimensions are validated before dispatch; the remaining indexed accesses follow those shape invariants"
)]

use rmlx_core::error::{Error, Result};
use rmlx_kv_quant::{KvCache, LinearAttnCache};
use rmlx_mlx::{add, concatenate, Array, Device};

use super::decoder_layer::{AttnBlock, MlpBlock};
use super::model::Qwen3_5MoeText;

fn batch_row(array: &Array, row: i32, device: Device) -> Result<Array> {
    let shape = array.shape();
    if shape.is_empty() || row < 0 || row >= shape[0] {
        return Err(Error::Model(format!(
            "Qwen3.5 batched decode: row {row} outside batch shape {shape:?}"
        )));
    }
    let mut starts = vec![0; shape.len()];
    let mut ends = shape.clone();
    let steps = vec![1; shape.len()];
    starts[0] = row;
    ends[0] = row + 1;
    array.slice(&starts, &ends, &steps, device)
}

fn concatenate_rows(rows: &[Array], device: Device) -> Result<Array> {
    if rows.is_empty() {
        return Err(Error::Model(
            "Qwen3.5 batched decode: cannot concatenate an empty row set".to_owned(),
        ));
    }
    let refs: Vec<&Array> = rows.iter().collect();
    concatenate(&refs, 0, device)
}

/// Merge one layer's per-request GatedDeltaNet states into a batch cache.
///
/// `None` means the active sequences have mismatched cache/tape states and
/// must use the row-wise fallback. This is intentionally fail-closed: making
/// up a zero state for only some rows would corrupt those sequences.
fn merge_linear_caches(
    caches: &[&mut LinearAttnCache],
    device: Device,
) -> Result<Option<LinearAttnCache>> {
    if caches.is_empty() || caches.iter().any(|cache| cache.tape.is_some()) {
        return Ok(None);
    }

    let merge_field =
        |select: fn(&LinearAttnCache) -> Option<&Array>| -> Result<Option<Option<Array>>> {
            let selected: Vec<Option<&Array>> = caches.iter().map(|cache| select(cache)).collect();
            if selected.iter().all(Option::is_none) {
                return Ok(Some(None));
            }
            if selected.iter().any(Option::is_none) {
                return Ok(None);
            }
            let rows: Vec<&Array> = selected.into_iter().flatten().collect();
            match concatenate(&rows, 0, device) {
                Ok(array) => Ok(Some(Some(array))),
                Err(_) => Ok(None),
            }
        };

    let Some(conv_state) = merge_field(|cache| cache.conv_state.as_ref())? else {
        return Ok(None);
    };
    let Some(delta_state) = merge_field(|cache| cache.delta_state.as_ref())? else {
        return Ok(None);
    };
    Ok(Some(LinearAttnCache {
        conv_state,
        delta_state,
        tape: None,
    }))
}

fn split_linear_cache(
    merged: LinearAttnCache,
    caches: &mut [&mut LinearAttnCache],
    device: Device,
) -> Result<()> {
    let batch = caches.len();
    let LinearAttnCache {
        conv_state,
        delta_state,
        tape,
    } = merged;
    if tape.is_some() {
        return Err(Error::Model(
            "Qwen3.5 batched decode: merged recurrent tape unexpectedly armed".to_owned(),
        ));
    }
    for (row, cache) in caches.iter_mut().enumerate() {
        let row_i32 = i32::try_from(row)
            .map_err(|_| Error::Model("Qwen3.5 batch exceeds i32 row range".to_owned()))?;
        cache.conv_state = conv_state
            .as_ref()
            .map(|state| batch_row(state, row_i32, device))
            .transpose()?;
        cache.delta_state = delta_state
            .as_ref()
            .map(|state| batch_row(state, row_i32, device))
            .transpose()?;
    }
    debug_assert_eq!(batch, caches.len());
    Ok(())
}

impl Qwen3_5MoeText {
    /// Decode one token for several active requests in one model call.
    ///
    /// `input_ids` is `[B, 1]`; each row has its own KV and linear-attention
    /// cache set. GatedDeltaNet and MLP/MoE work is batched across B. Full
    /// attention projects/norms the hidden rows together, then updates and
    /// attends each request's KV cache independently so its offset and chosen
    /// KV codec remain exactly the single-stream path.
    ///
    /// When a GatedDeltaNet row has a tape armed (speculative generation), or
    /// recurrent caches have mixed initialization state, the method falls
    /// back to independent single-row forwards. That fallback is the
    /// correctness reference; it is not silently treated as a fused batch.
    pub fn forward_batch_arr(
        &self,
        input_ids: &Array,
        kv_cache_rows: &mut [&mut [KvCache]],
        lin_cache_rows: &mut [&mut [LinearAttnCache]],
        device: Device,
    ) -> Result<Array> {
        let shape = input_ids.shape();
        if shape.len() != 2 || shape[1] != 1 {
            return Err(Error::Model(format!(
                "Qwen3.5 batched decode expects input_ids [B, 1], got {shape:?}"
            )));
        }
        let batch = usize::try_from(shape[0])
            .map_err(|_| Error::Model("Qwen3.5 batch has negative size".to_owned()))?;
        if batch == 0 || kv_cache_rows.len() != batch || lin_cache_rows.len() != batch {
            return Err(Error::Model(format!(
                "Qwen3.5 batched decode cache rows must equal nonzero batch size {batch} (kv={}, linear={})",
                kv_cache_rows.len(),
                lin_cache_rows.len()
            )));
        }
        let n_layers = self.layers.len();
        if kv_cache_rows.iter().any(|row| row.len() != n_layers)
            || lin_cache_rows.iter().any(|row| row.len() != n_layers)
        {
            return Err(Error::Model(format!(
                "Qwen3.5 batched decode requires {n_layers} KV and linear caches per row"
            )));
        }

        // An armed tape or partially initialized GDN state cannot be merged
        // without changing the speculative rollback contract. Fall back to the
        // single-sequence forward for every row in that case.
        let has_tape = lin_cache_rows
            .iter()
            .any(|row| row.iter().any(|cache| cache.tape.is_some()));
        if has_tape {
            return self.forward_rows_sequential(input_ids, kv_cache_rows, lin_cache_rows, device);
        }

        // Merge every recurrent layer before mutating any model/cache state.
        // A mixed/shape-incompatible state therefore takes the sequential
        // fallback from the untouched caches, never halfway through a stack.
        let mut merged_linear = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            if matches!(self.layers[layer_idx].attn, AttnBlock::Linear(_)) {
                let row_refs: Vec<&mut LinearAttnCache> = lin_cache_rows
                    .iter_mut()
                    .map(|row| &mut row[layer_idx])
                    .collect();
                let Some(merged) = merge_linear_caches(&row_refs, device)? else {
                    return self.forward_rows_sequential(
                        input_ids,
                        kv_cache_rows,
                        lin_cache_rows,
                        device,
                    );
                };
                merged_linear.push(Some(merged));
            } else {
                merged_linear.push(None);
            }
        }

        let mut hidden = self.embed_tokens.forward(input_ids, device)?;
        hidden = hidden.reshape(&[shape[0], 1, self.cfg.hidden_size as i32], device)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = hidden.try_clone()?;
            let normed = layer.input_layernorm.forward(&hidden, device)?;
            let attended = match &layer.attn {
                AttnBlock::Linear(gdn) => {
                    let merged = merged_linear[layer_idx]
                        .as_mut()
                        .expect("linear-attention layer has its merged recurrent cache");
                    gdn.forward(&normed, Some(merged), device)?
                }
                AttnBlock::Full(full) => {
                    // FullAttention owns each request's growing KV sequence
                    // and offset. Keep this stateful part row-wise; the input
                    // norm and subsequent MLP remain batched.
                    let mut rows = Vec::with_capacity(batch);
                    for row in 0..batch {
                        let row_i32 = i32::try_from(row).map_err(|_| {
                            Error::Model("Qwen3.5 batch exceeds i32 row range".to_owned())
                        })?;
                        let x = batch_row(&normed, row_i32, device)?;
                        let offset = kv_cache_rows[row][layer_idx].offset();
                        rows.push(full.forward(
                            &x,
                            offset,
                            Some(&mut kv_cache_rows[row][layer_idx]),
                            None,
                            device,
                        )?);
                    }
                    concatenate_rows(&rows, device)?
                }
            };
            hidden = add(&residual, &attended, device)?;

            let residual = hidden.try_clone()?;
            let normed = layer.post_attention_layernorm.forward(&hidden, device)?;
            let mlp = match &layer.mlp {
                MlpBlock::Moe(moe) => moe.forward(&normed, device)?,
                MlpBlock::Dense(dense) => dense.forward(&normed, device)?,
            };
            hidden = add(&residual, &mlp, device)?;
        }

        let hidden = self.final_norm.forward(&hidden, device)?;
        let logits = match &self.lm_head {
            Some(head) => head.forward(&hidden, device)?,
            None => self.embed_tokens.as_linear(&hidden, device)?,
        };
        for (layer_idx, merged) in merged_linear.into_iter().enumerate() {
            let Some(merged) = merged else {
                continue;
            };
            let mut row_refs: Vec<&mut LinearAttnCache> = lin_cache_rows
                .iter_mut()
                .map(|row| &mut row[layer_idx])
                .collect();
            split_linear_cache(merged, &mut row_refs, device)?;
        }
        Ok(logits)
    }

    fn forward_rows_sequential(
        &self,
        input_ids: &Array,
        kv_cache_rows: &mut [&mut [KvCache]],
        lin_cache_rows: &mut [&mut [LinearAttnCache]],
        device: Device,
    ) -> Result<Array> {
        let batch = input_ids.shape()[0] as usize;
        let mut rows = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_i32 = i32::try_from(row)
                .map_err(|_| Error::Model("Qwen3.5 batch exceeds i32 row range".to_owned()))?;
            let ids = batch_row(input_ids, row_i32, device)?;
            rows.push(self.forward_arr(
                &ids,
                1,
                Some(&mut kv_cache_rows[row]),
                Some(&mut lin_cache_rows[row]),
                device,
            )?);
        }
        concatenate_rows(&rows, device)
    }
}
