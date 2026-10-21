use super::compatible;
use crate::engine::{GenerationRequest, SamplingParams};

fn request() -> GenerationRequest {
    GenerationRequest {
        model_id: "qwen-coder".to_owned(),
        prompt_tokens: vec![1, 2, 3],
        max_tokens: 8,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        stop: vec![],
        stream: true,
        system: None,
        session_id: None,
        effective_prompt_cache_slots: None,
        metrics_drainer: None,
        itl_store: None,
        event_recorder: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        constraint: None,
        is_thinking_handle: None,
        thinking_budget: None,
        thinking_end_token_id: None,
        prompt_think_open: false,
        emit_tool_markers: false,
        thinking_start_token: None,
        thinking_end_token: None,
        gpu_admission: None,
        kv_quant_override: None,
        max_ctx_override: None,
        images: vec![],
        audio_b64: vec![],
        image_max_tokens: None,
    }
}

#[test]
fn supported_text_request_is_batch_compatible() {
    assert!(compatible(&request()));
}

#[test]
fn sampling_and_penalties_keep_independent_row_state() {
    let mut req = request();
    req.sampling.temperature = 0.2;
    req.sampling.top_p = 0.9;
    req.sampling.top_k = 40;
    req.sampling.min_p = 0.05;
    req.sampling.seed = Some(7);
    req.sampling.repetition_penalty = 1.1;
    req.sampling.frequency_penalty = 0.2;
    req.sampling.presence_penalty = 0.1;
    req.sampling.logit_bias.push((42, 0.5));
    assert!(compatible(&req));
}

#[test]
fn logprob_capture_falls_back_until_batched_logprob_rows_are_implemented() {
    let mut req = request();
    req.sampling.top_logprobs_k = 1;
    assert!(!compatible(&req));
}

#[test]
fn constraints_tools_and_multimodal_requests_fall_back() {
    let mut req = request();
    req.response_format = Some(crate::engine::NormalizedResponseFormat::JsonObject);
    assert!(!compatible(&req));

    let mut req = request();
    req.audio_b64.push("synthetic".to_owned());
    assert!(!compatible(&req));

    let mut req = request();
    req.stop.push("STOP".to_owned());
    assert!(!compatible(&req));
}

#[test]
fn prompt_context_over_batcher_limit_falls_back() {
    let mut req = request();
    req.prompt_tokens
        .resize(super::MAX_BATCH_PROMPT_TOKENS + 1, 1);
    assert!(!compatible(&req));
}
