//! Tests for [`super::estimate`] and [`super::run_estimate_memory`].
//!
//! All tests build a synthetic snapshot directory under `tempfile::tempdir`
//! (a real `config.json` + `model.safetensors.index.json`, no actual tensor
//! bytes needed — the estimator only reads declared sizes and config
//! fields) so none of this depends on `RMLX_O_MODELS_ROOT` or a real model.

use std::path::Path;

use rmlx_kv_quant::KvQuant;

use super::{count_kv_layers, estimate, resolve_weights_bytes};

/// Write a minimal multi-shard-index snapshot: `config.json` +
/// `model.safetensors.index.json` with `metadata.total_size` set, no actual
/// `.safetensors` payload (the estimator never opens the shard bytes).
fn write_hybrid_snapshot(dir: &Path, total_size: u64) {
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 8,
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "max_position_embeddings": 4096,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ]
        },
        "quantization": {"bits": 4, "group_size": 64}
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    let index = serde_json::json!({
        "metadata": {"total_size": total_size},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00001.safetensors"
        }
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
}

/// Uniform-attention snapshot (no `layer_types`): every layer should count
/// as KV-holding.
fn write_dense_snapshot(dir: &Path, total_size: u64) {
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "text_config": {
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 4096
        },
        "quantization": {"bits": 4, "group_size": 64}
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    let index = serde_json::json!({
        "metadata": {"total_size": total_size},
        "weight_map": {
            "model.layers.0.weight": "model.safetensors"
        }
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
}

#[test]
fn hybrid_snapshot_counts_only_full_attention_layers_for_kv() {
    let dir = tempfile::tempdir().unwrap();
    write_hybrid_snapshot(dir.path(), 17_000_000_000);

    let est = estimate(dir.path(), 16_384, KvQuant::None).expect("estimate should succeed");

    // 8 layers declared, only 2 are "full_attention" -> only 2 hold KV.
    assert_eq!(est.total_layers, 8);
    assert_eq!(est.kv_layers_counted, 2);

    // weights_bytes comes straight from metadata.total_size.
    assert_eq!(est.weights_bytes, 17_000_000_000);

    // KV bytes for KvQuant::None at seq=16384, head_dim=64, kv_heads=2:
    // elems = 16384 * 64 * 2 = 2_097_152; bf16 K+V = elems * 2 * 2.
    let elems: u64 = 16_384 * 64 * 2;
    let per_layer = elems * 2 * 2;
    assert_eq!(est.kv_bytes_total, per_layer * 2); // 2 kv-holding layers

    assert_eq!(
        est.total_bytes,
        est.weights_bytes + est.kv_bytes_total + est.activation_bytes
    );
}

#[test]
fn dense_snapshot_without_layer_types_counts_every_layer() {
    let dir = tempfile::tempdir().unwrap();
    write_dense_snapshot(dir.path(), 500_000_000);

    let est = estimate(dir.path(), 4096, KvQuant::None).expect("estimate should succeed");

    assert_eq!(est.total_layers, 4);
    assert_eq!(est.kv_layers_counted, 4); // no layer_types -> every layer counts
}

#[test]
fn fits_is_false_when_total_exceeds_budget() {
    let dir = tempfile::tempdir().unwrap();
    // Declare an absurdly large weights size so this fails on any real host.
    write_dense_snapshot(dir.path(), 100 * 1024 * 1024 * 1024 * 1024); // 100 TiB

    let est = estimate(dir.path(), 4096, KvQuant::None).expect("estimate should succeed");
    assert!(!est.fits, "100 TiB of weights must not fit any real host");
    assert!(est.total_bytes > est.budget_bytes);
}

#[test]
fn missing_num_hidden_layers_is_an_error_not_a_silent_zero() {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "text_config": {
            "hidden_size": 256,
            "num_attention_heads": 4,
            "head_dim": 64
        }
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    let index = serde_json::json!({
        "metadata": {"total_size": 1_000_000},
        "weight_map": {"model.layers.0.weight": "model.safetensors"}
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();

    let result = estimate(dir.path(), 4096, KvQuant::None);
    assert!(
        result.is_err(),
        "a config with no num_hidden_layers must error, not silently report kv_bytes_total=0"
    );
}

#[test]
fn weights_bytes_falls_back_to_shard_file_sizes_without_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64
        }
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();

    // No `metadata` block at all -> forces the shard-file-size fallback.
    let index = serde_json::json!({
        "weight_map": {"model.layers.0.weight": "model.safetensors"}
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();
    let payload = vec![0_u8; 12_345];
    std::fs::write(dir.path().join("model.safetensors"), &payload).unwrap();

    let est = estimate(dir.path(), 128, KvQuant::None).expect("estimate should succeed");
    assert_eq!(est.weights_bytes, 12_345);
}

#[test]
fn count_kv_layers_handles_a_config_with_no_text_config() {
    let json = serde_json::json!({"architectures": ["Qwen3ForCausalLM"]});
    let cfg: rmlx_loader::ModelConfig = serde_json::from_value(json).unwrap();
    let (kv_layers, total_layers) = count_kv_layers(&cfg);
    assert_eq!(kv_layers, 0);
    assert_eq!(total_layers, 0);
}

/// Writes a minimal valid `.safetensors` file: an 8-byte little-endian
/// header length, the JSON header (one `F32` tensor, no `__metadata__`
/// needed), then that many zero bytes of tensor payload.
fn write_minimal_safetensors(path: &Path, payload_len: usize) {
    let header = serde_json::json!({
        "model.layers.0.weight": {
            "dtype": "F32",
            "shape": [1, payload_len / 4],
            "data_offsets": [0, payload_len]
        }
    });
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    // Pad to 8-byte alignment, as the safetensors format requires.
    let pad = (8 - header_bytes.len() % 8) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', pad));

    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend(std::iter::repeat_n(0_u8, payload_len));
    std::fs::write(path, out).unwrap();
}

/// This exercises the real [`rmlx_loader::load_shard_index`] path (not a
/// hand-built `ShardIndex`), which builds the synthetic single-file index by
/// **parsing the safetensors header off disk** — a case the other
/// `resolve_weights_bytes_*` test below bypasses by constructing a
/// `ShardIndex` directly. A single-shard snapshot's synthetic index carries
/// no `metadata.total_size`, so this is also the real-world path that
/// exercises the shard-file-size fallback end to end.
#[test]
fn single_shard_snapshot_with_no_index_file_resolves_via_safetensors_header() {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64
        }
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    // No model.safetensors.index.json at all -- forces load_shard_index's
    // synthetic single-file path, which opens and parses this file's real
    // safetensors header.
    write_minimal_safetensors(&dir.path().join("model.safetensors"), 64);

    let est = estimate(dir.path(), 4096, KvQuant::None).expect("estimate should succeed");
    let actual_file_size = std::fs::metadata(dir.path().join("model.safetensors"))
        .unwrap()
        .len();
    assert_eq!(est.weights_bytes, actual_file_size);
    assert_eq!(est.kv_layers_counted, 2);
}

#[test]
fn resolve_weights_bytes_prefers_metadata_total_size_over_shard_scan() {
    let dir = tempfile::tempdir().unwrap();
    // A shard file exists on disk with a different size than the declared
    // metadata -- the function must prefer the declared total_size.
    std::fs::write(dir.path().join("model.safetensors"), vec![0_u8; 999]).unwrap();
    let mut weight_map = std::collections::BTreeMap::new();
    weight_map.insert(
        "model.layers.0.weight".to_owned(),
        "model.safetensors".to_owned(),
    );
    let idx = rmlx_loader::ShardIndex {
        metadata: serde_json::json!({"total_size": 123_456}),
        weight_map,
    };
    let bytes = resolve_weights_bytes(dir.path(), &idx).unwrap();
    assert_eq!(
        bytes, 123_456,
        "declared total_size must win over actual file size"
    );
}
