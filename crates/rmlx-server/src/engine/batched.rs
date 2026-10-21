//! Experimental continuous batching for compatible Qwen3.5-MoE text requests.
//!
//! A short initial admission window groups the first requests. Between decode
//! ticks, newly admitted requests can join and finished/cancelled rows leave;
//! every request retains its independent KV/recurrent cache and sampler state.
//! Unsupported features remain on the reference serial path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use futures::{stream, Stream};
use parking_lot::Mutex;
use rmlx_core::Error;
use rmlx_kv_quant::{KvCache, LinearAttnCache};
use rmlx_mlx::{Array, Device, Dtype};

use super::arch_generator::ArchGenerator;
use super::think::ThinkSplitter;
use super::types::{GenerationRequest, GenerationToken};
use crate::detokenizer::StreamingDetokenizer;
use crate::engine::helpers::terminal_item;

const BATCH_WINDOW: Duration = Duration::from_millis(12);
const MAX_BATCH_PROMPT_TOKENS: usize = 2048;

struct Pending {
    req: GenerationRequest,
    tx: tokio::sync::mpsc::Sender<rmlx_core::Result<GenerationToken>>,
}

/// Owner of the worker queue. Dropping the last sender lets the worker exit.
pub(crate) struct ContinuousBatchScheduler {
    tx: mpsc::Sender<Pending>,
    last_kv_bytes: Arc<AtomicU64>,
}

impl ContinuousBatchScheduler {
    pub(crate) fn attach(generator: &ArchGenerator, batch_size: usize) -> Option<Arc<Self>> {
        if batch_size < 2 || !generator.model.supports_batched_decode() {
            return None;
        }

        let (tx, rx) = mpsc::channel::<Pending>();
        let model = Arc::clone(&generator.model);
        let tokenizer = Arc::clone(&generator.tokenizer);
        let eos_ids = Arc::clone(&generator.eos_ids);
        let model_id = generator.model_id.clone();
        let device = generator.device;
        let kv_quant = generator
            .kv_quant_override
            .unwrap_or(rmlx_models::kv_cache::DEFAULT_KV_QUANT);
        let max_ctx = generator.max_ctx_override;
        let tokenizer_kind = generator.tokenizer_kind;
        let effective_max_ctx = generator.effective_max_ctx;
        let lock = Arc::clone(&generator._lock);
        let last_kv_bytes = Arc::new(AtomicU64::new(0));
        let worker_last_kv_bytes = Arc::clone(&last_kv_bytes);

        std::thread::Builder::new()
            .name(format!("rmlx-batch-{model_id}"))
            .spawn(move || {
                worker_loop(
                    rx,
                    batch_size,
                    model,
                    tokenizer,
                    eos_ids,
                    device,
                    kv_quant,
                    max_ctx,
                    tokenizer_kind,
                    lock,
                    worker_last_kv_bytes,
                    effective_max_ctx,
                );
            })
            .ok()?;

        Some(Arc::new(Self { tx, last_kv_bytes }))
    }

    pub(crate) fn last_kv_cache_bytes(&self) -> u64 {
        self.last_kv_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn submit(&self, req: GenerationRequest) -> PinBoxTokenStream {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        if self.tx.send(Pending { req, tx }).is_err() {
            return Box::pin(stream::once(async {
                Err(Error::Other(
                    "batched generation worker has stopped".to_owned(),
                ))
            }));
        }
        Box::pin(stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }))
    }
}

type PinBoxTokenStream =
    std::pin::Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>;

/// Requests enter the batcher only when their semantics are exactly within
/// the supported text subset. All other requests use the reference engine.
pub(crate) fn compatible(req: &GenerationRequest) -> bool {
    let s = &req.sampling;
    req.max_tokens > 0
        && !req.prompt_tokens.is_empty()
        && req.prompt_tokens.len() <= MAX_BATCH_PROMPT_TOKENS
        && s.top_logprobs_k == 0
        && req.stop.is_empty()
        && req.constraint.is_none()
        && req.tools.is_none()
        && req.tool_choice.is_none()
        && req.response_format.is_none()
        && req.images.is_empty()
        && req.audio_b64.is_empty()
        && req.session_id.is_none()
        && req.effective_prompt_cache_slots.is_none()
        && req.is_thinking_handle.is_none()
        && req.thinking_end_token_id.is_none()
        && req.image_max_tokens.is_none()
        && req.kv_quant_override.is_none()
        && req.max_ctx_override.is_none()
        && req.thinking_budget.is_none()
        && req.thinking_start_token.is_none()
        && req.thinking_end_token.is_none()
        && !req.emit_tool_markers
}

struct Row {
    pending: Pending,
    kv: Vec<KvCache>,
    linear: Vec<LinearAttnCache>,
    detok: StreamingDetokenizer,
    splitter: Option<ThinkSplitter>,
    timestamps: Vec<std::time::Instant>,
    generated: usize,
    last_id: u32,
    sampler: rmlx_models::SamplerConfig,
    penalty: rmlx_models::PenaltyConfig,
    rng: rmlx_models::Pcg32,
    token_history: Vec<u32>,
    last_kv_bytes: Arc<AtomicU64>,
    kv_quant: rmlx_kv_quant::KvQuant,
    ctx_max: i64,
}

#[allow(
    clippy::too_many_arguments,
    clippy::needless_pass_by_value,
    clippy::match_same_arms,
    reason = "the worker thread owns these resources for its full lifetime; identical receive errors both end the current cohort window"
)]
fn worker_loop(
    rx: mpsc::Receiver<Pending>,
    batch_size: usize,
    model: Arc<rmlx_models::arch::Architecture>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    eos_ids: Arc<Vec<u32>>,
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    tokenizer_kind: crate::detokenizer::TokenizerKind,
    lock: Arc<Mutex<()>>,
    last_kv_bytes: Arc<AtomicU64>,
    effective_max_ctx: usize,
) {
    while let Ok(first) = rx.recv() {
        let mut cohort = vec![first];
        let deadline = std::time::Instant::now() + BATCH_WINDOW;
        while cohort.len() < batch_size {
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                break;
            };
            match rx.recv_timeout(left) {
                Ok(pending) => cohort.push(pending),
                Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                    break
                }
            }
        }
        let failure_senders: Vec<_> = cohort.iter().map(|pending| pending.tx.clone()).collect();
        let _guard = lock.lock();
        if let Err(e) = run_cohort(
            cohort,
            &rx,
            batch_size,
            &model,
            &tokenizer,
            &eos_ids,
            device,
            kv_quant,
            max_ctx_override,
            tokenizer_kind,
            &last_kv_bytes,
            i64::try_from(effective_max_ctx).unwrap_or(i64::MAX),
        ) {
            tracing::error!(error = %e, "Qwen3.5 continuous batch scheduler failed");
            for tx in failure_senders {
                let _ =
                    tx.blocking_send(Err(Error::Other(format!("batched generation failed: {e}"))));
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run_cohort(
    cohort: Vec<Pending>,
    rx: &mpsc::Receiver<Pending>,
    batch_size: usize,
    model: &rmlx_models::arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    eos_ids: &[u32],
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    tokenizer_kind: crate::detokenizer::TokenizerKind,
    last_kv_bytes: &Arc<AtomicU64>,
    ctx_max: i64,
) -> rmlx_core::Result<()> {
    let limits = model.context_limits();
    let ctx = rmlx_models::context::resolve_context(&limits, max_ctx_override)?;
    let n_layers = model.num_hidden_layers();
    let layer_quants =
        rmlx_models::kv_cache::kv_layer_quants(n_layers, kv_quant, model.shares_kv_across_layers());
    let initial_cohort_size = cohort.len();
    let mut rows = Vec::with_capacity(batch_size);
    let mut joined_rows = 0_usize;
    let mut decode_calls = 0_u64;
    let mut multirow_decode_calls = 0_u64;
    let mut max_active_rows = 0_usize;

    for pending in cohort {
        if let Some(row) = prepare_row(
            pending,
            model,
            tokenizer,
            eos_ids,
            device,
            &ctx,
            &layer_quants,
            tokenizer_kind,
            Arc::clone(last_kv_bytes),
            kv_quant,
            ctx_max,
        ) {
            rows.push(row);
        }
    }

    while !rows.is_empty() {
        rows.retain(|row| !row.pending.tx.is_closed());
        while rows.len() < batch_size {
            match rx.try_recv() {
                Ok(pending) => {
                    if let Some(row) = prepare_row(
                        pending,
                        model,
                        tokenizer,
                        eos_ids,
                        device,
                        &ctx,
                        &layer_quants,
                        tokenizer_kind,
                        Arc::clone(last_kv_bytes),
                        kv_quant,
                        ctx_max,
                    ) {
                        rows.push(row);
                        joined_rows += 1;
                    }
                }
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => break,
            }
        }
        if rows.is_empty() {
            break;
        }
        max_active_rows = max_active_rows.max(rows.len());
        decode_calls += 1;
        if rows.len() > 1 {
            multirow_decode_calls += 1;
        }
        let ids: Vec<i32> = rows.iter().map(|row| row.last_id as i32).collect();
        let batch = i32::try_from(ids.len())
            .map_err(|_| Error::InvalidInput("batch size exceeds i32".to_owned()))?;
        let input = Array::from_i32_slice(&ids, &[batch, 1])?;
        let mut kv_rows: Vec<&mut [KvCache]> = Vec::with_capacity(rows.len());
        let mut linear_rows: Vec<&mut [LinearAttnCache]> = Vec::with_capacity(rows.len());
        for row in &mut rows {
            kv_rows.push(row.kv.as_mut_slice());
            linear_rows.push(row.linear.as_mut_slice());
        }
        let logits =
            match model.forward_batched_decode(&input, &mut kv_rows, &mut linear_rows, device) {
                Ok(logits) => logits,
                Err(error) => {
                    fail_rows(&rows, &error);
                    return Ok(());
                }
            };
        let tokens = match select_rows(&logits, &mut rows, device) {
            Ok(tokens) => tokens,
            Err(error) => {
                fail_rows(&rows, &error);
                return Ok(());
            }
        };
        let mut index = 0;
        rows.retain_mut(|row| {
            let Some(&token) = tokens.get(index) else {
                let _ = row.pending.tx.blocking_send(Err(Error::Model(
                    "batched decode returned too few token rows".to_owned(),
                )));
                index += 1;
                return false;
            };
            index += 1;
            row.last_id = token;
            if let Err(error) = emit_token(row, tokenizer, token) {
                let _ = row.pending.tx.blocking_send(Err(error));
                return false;
            }
            if eos_ids.contains(&token) || row.generated >= row.pending.req.max_tokens as usize {
                finish_row(row, tokenizer, eos_ids);
                false
            } else {
                true
            }
        });
    }
    tracing::info!(
        initial_cohort_size,
        joined_rows,
        decode_calls,
        multirow_decode_calls,
        max_active_rows,
        "Qwen3.5 batched cohort complete"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_row(
    pending: Pending,
    model: &rmlx_models::arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    eos_ids: &[u32],
    device: Device,
    ctx: &rmlx_models::context::ResolvedContext,
    layer_quants: &[rmlx_kv_quant::KvQuant],
    tokenizer_kind: crate::detokenizer::TokenizerKind,
    last_kv_bytes: Arc<AtomicU64>,
    kv_quant: rmlx_kv_quant::KvQuant,
    ctx_max: i64,
) -> Option<Row> {
    if pending.tx.is_closed() {
        return None;
    }
    let n_layers = model.num_hidden_layers();
    let mut kv: Vec<KvCache> = layer_quants
        .iter()
        .enumerate()
        .map(|(layer, &quant)| {
            KvCache::with_quant_max_seq(quant, ctx.initial_max_seq)
                .with_max_seq_ceiling(ctx.ceiling)
                .with_layer_idx(layer)
        })
        .collect();
    let mut linear: Vec<LinearAttnCache> = (0..n_layers).map(|_| LinearAttnCache::new()).collect();
    for cache in &mut kv {
        cache.enter_prefill();
    }
    let prefill = model.forward_seq_last_k_with_cache(
        &pending.req.prompt_tokens,
        1,
        &mut kv,
        Some(&mut linear),
        device,
    );
    let logits = match prefill {
        Ok(logits) => logits,
        Err(error) => {
            let _ = pending.tx.blocking_send(Err(error));
            return None;
        }
    };
    let mut exit_error = None;
    for cache in &mut kv {
        if let Err(error) = cache.exit_prefill(device) {
            exit_error.get_or_insert(error);
        }
    }
    if let Some(error) = exit_error {
        let _ = pending.tx.blocking_send(Err(error));
        return None;
    }
    if !matches!(logits.dtype(), Dtype::F32 | Dtype::Bf16) {
        let _ = pending.tx.blocking_send(Err(Error::Other(format!(
            "Qwen3.5 batched prefill logits use unreadable dtype {:?}; refusing unverified output",
            logits.dtype()
        ))));
        return None;
    }
    let prefill_bytes = match logits.to_bytes() {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = pending.tx.blocking_send(Err(error));
            return None;
        }
    };
    if rmlx_runtime::count_nan_in_bytes(&prefill_bytes, logits.dtype()) > 0 {
        let _ = pending.tx.blocking_send(Err(Error::Other(
            "Qwen3.5 batched prefill logits contain NaN; refusing to generate".to_owned(),
        )));
        return None;
    }
    let sampler = rmlx_models::SamplerConfig {
        temperature: pending.req.sampling.temperature,
        top_p: pending.req.sampling.top_p,
        top_k: pending.req.sampling.top_k,
        min_p: pending.req.sampling.min_p,
        seed: pending.req.sampling.seed,
        top_logprobs_k: pending.req.sampling.top_logprobs_k,
    };
    let penalty = rmlx_models::PenaltyConfig {
        rep_penalty: pending.req.sampling.repetition_penalty,
        presence_penalty: pending.req.sampling.presence_penalty,
        frequency_penalty: pending.req.sampling.frequency_penalty,
        logit_bias: pending.req.sampling.logit_bias.clone(),
    };
    let mut rng = rmlx_models::Pcg32::new(sampler.seed_or_default());
    let token = match select_one(&logits, &sampler, &penalty, &[], &mut rng, device) {
        Ok(token) => token,
        Err(error) => {
            let _ = pending.tx.blocking_send(Err(error));
            return None;
        }
    };
    let mut row = Row {
        pending,
        kv,
        linear,
        detok: StreamingDetokenizer::new(tokenizer_kind, Arc::new(eos_ids.to_vec())),
        splitter: None,
        timestamps: Vec::new(),
        generated: 0,
        last_id: token,
        sampler,
        penalty,
        rng,
        token_history: Vec::new(),
        last_kv_bytes,
        kv_quant,
        ctx_max,
    };
    row.splitter = Some(ThinkSplitter::new_for_request(
        row.pending.req.prompt_think_open,
        None,
        None,
        None,
    ));
    if let Err(error) = emit_token(&mut row, tokenizer, token) {
        if !matches!(error, Error::Cancelled) {
            let _ = row.pending.tx.blocking_send(Err(error));
        }
        return None;
    }
    let limit = row.pending.req.max_tokens as usize;
    if eos_ids.contains(&token) || row.generated >= limit {
        finish_row(&mut row, tokenizer, eos_ids);
        None
    } else {
        Some(row)
    }
}

fn fail_rows(rows: &[Row], error: &Error) {
    let message = format!("batched request cohort failed: {error}");
    for row in rows {
        let _ = row
            .pending
            .tx
            .blocking_send(Err(Error::Other(message.clone())));
    }
}

fn select_one(
    logits: &Array,
    sampler: &rmlx_models::SamplerConfig,
    penalty: &rmlx_models::PenaltyConfig,
    history: &[u32],
    rng: &mut rmlx_models::Pcg32,
    device: Device,
) -> rmlx_core::Result<u32> {
    let shape = logits.shape();
    let vocab = *shape
        .last()
        .ok_or_else(|| Error::Model("empty sampler logits".to_owned()))?;
    let logits_row = logits.reshape(&[1, vocab], device)?;
    let selected = rmlx_models::sampler::select_token(
        &logits_row,
        sampler,
        None,
        penalty,
        history,
        rng,
        device,
    )?;
    selected.eval()?;
    let bytes = selected.to_bytes()?;
    let token_bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| {
        Error::Model(format!(
            "sampler returned {} bytes, expected 4",
            bytes.len()
        ))
    })?;
    Ok(u32::from_le_bytes(token_bytes))
}

fn select_rows(logits: &Array, rows: &mut [Row], device: Device) -> rmlx_core::Result<Vec<u32>> {
    let shape = logits.shape();
    let batch = i32::try_from(rows.len())
        .map_err(|_| Error::InvalidInput("batch size exceeds i32".to_owned()))?;
    let vocab = match shape.as_slice() {
        [shape_batch, 1, vocab] if *shape_batch == batch => *vocab,
        _ => {
            return Err(Error::Model(format!(
                "batched logits shape {shape:?} does not match {batch} request rows"
            )));
        }
    };
    let flat = logits.reshape(&[batch, vocab], device)?;
    let mut tokens = Vec::with_capacity(rows.len());
    for (idx, row) in rows.iter_mut().enumerate() {
        let row_i32 = i32::try_from(idx)
            .map_err(|_| Error::InvalidInput("batch row exceeds i32".to_owned()))?;
        let logits_row = flat.slice(&[row_i32, 0], &[row_i32 + 1, vocab], &[1, 1], device)?;
        let recent_start = row.token_history.len().saturating_sub(20);
        let recent = row.token_history.get(recent_start..).ok_or_else(|| {
            Error::Model("request token-history window is out of bounds".to_owned())
        })?;
        tokens.push(select_one(
            &logits_row,
            &row.sampler,
            &row.penalty,
            recent,
            &mut row.rng,
            device,
        )?);
    }
    Ok(tokens)
}

fn emit_token(
    row: &mut Row,
    tokenizer: &tokenizers::Tokenizer,
    token_id: u32,
) -> rmlx_core::Result<()> {
    row.generated += 1;
    row.token_history.push(token_id);
    row.timestamps.push(std::time::Instant::now());
    let text = row
        .detok
        .step(tokenizer, token_id)
        .map_err(|error| Error::Other(format!("batched detokenization failed: {error}")))?;
    let (piece, is_thinking) = row
        .splitter
        .as_mut()
        .map_or((text.clone(), false), |splitter| splitter.step(&text));
    row.pending
        .tx
        .blocking_send(Ok(GenerationToken {
            token_id,
            piece,
            done: false,
            finish_reason: None,
            is_thinking,
            logprobs: None,
        }))
        .map_err(|_| Error::Cancelled)
}

fn finish_row(row: &mut Row, tokenizer: &tokenizers::Tokenizer, eos_ids: &[u32]) {
    let finish = if eos_ids.contains(&row.last_id) {
        "stop"
    } else {
        "length"
    };
    let kv_bytes = row.kv.iter().map(KvCache::resident_bytes).sum::<u64>()
        + row
            .linear
            .iter()
            .map(LinearAttnCache::resident_bytes)
            .sum::<u64>();
    row.last_kv_bytes.store(kv_bytes, Ordering::Relaxed);
    let model_id = row.pending.req.model_id.clone();
    let quant_label = rmlx_kv_quant::kv_quant_label(Some(row.kv_quant));
    let ts = super::helpers::spsc_ts();
    if let Some(drainer) = row.pending.req.metrics_drainer.as_ref() {
        use crate::metrics_drainer::{MetricEvent, MetricKind};
        drainer.try_emit(MetricEvent {
            model_id: model_id.clone(),
            kv_quant: quant_label.clone(),
            ts_utc: ts.clone(),
            ctx_max: row.ctx_max,
            kind: MetricKind::KvCacheBytes(kv_bytes),
        });
    }
    if kv_bytes > 0 {
        if let Some(recorder) = row.pending.req.event_recorder.as_ref() {
            if let Err(error) = recorder.record(&rmlx_metrics::events::Measurement {
                model_path: &model_id,
                quant_mode: &quant_label,
                stage: "request",
                op: "kv_cache_bytes",
                value_unit: "bytes",
                value: kv_bytes as f64,
                notes: "continuous batched decode row",
            }) {
                tracing::warn!(error = %error, "batched KV bytes event write failed");
            }
        }
    }
    if let Some((p50, p95, p99, mean, spikes)) = super::helpers::compute_itl_stats(&row.timestamps)
    {
        if let Some(store) = row.pending.req.itl_store.as_ref() {
            let mut ring = store.lock();
            if ring.len() >= crate::openai::ITL_RING_CAPACITY {
                ring.pop_front();
            }
            ring.push_back(crate::openai::ItlSample {
                model_id: model_id.clone(),
                p50_ms: p50,
                p95_ms: p95,
                mean_ms: mean,
                step_count: row.timestamps.len(),
            });
        }
        if let Some(drainer) = row.pending.req.metrics_drainer.as_ref() {
            use crate::metrics_drainer::{MetricEvent, MetricKind};
            for kind in [
                MetricKind::ItlStats {
                    p50_ms: p50,
                    p95_ms: p95,
                    mean_ms: mean,
                    step_count: row.timestamps.len(),
                },
                MetricKind::ItlP99Ms(p99),
                MetricKind::ItlSpikes(spikes),
            ] {
                drainer.try_emit(MetricEvent {
                    model_id: model_id.clone(),
                    kv_quant: quant_label.clone(),
                    ts_utc: ts.clone(),
                    ctx_max: row.ctx_max,
                    kind,
                });
            }
        }
        if let Some(recorder) = row.pending.req.event_recorder.as_ref() {
            super::helpers::record_itl_percentiles(
                recorder,
                &model_id,
                &quant_label,
                p50,
                p95,
                p99,
            );
        }
    }
    let terminal = terminal_item(&mut row.detok, tokenizer, row.last_id, finish.to_owned());
    let _ = row.pending.tx.blocking_send(Ok(terminal));
}

#[cfg(test)]
#[path = "batched_tests.rs"]
mod batched_tests;
