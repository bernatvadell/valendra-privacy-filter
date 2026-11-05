"""Safe, traceable weight-only initialization from a prior Qwen3 experiment."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from valendra_privacy_filter.datasets.prepare import _sha256_file
from valendra_privacy_filter.models import PrivacyFilterModel
from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION


ARCHITECTURE_FIELDS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "rope_theta", "rms_norm_eps", "hidden_act", "max_position_embeddings",
)


def checkpoint_provenance(checkpoint, target_config, names, attention_mode):
    checkpoint = Path(checkpoint).resolve()
    weights, configuration = checkpoint / "model.safetensors", checkpoint / "config.json"
    state_file, identity_file = checkpoint / "trainer_state.json", checkpoint.parent / "run_identity.json"
    if not all(path.is_file() for path in (weights, configuration, state_file, identity_file)):
        raise ValueError("Warm-start checkpoint must contain Safetensors, configuration, Trainer state and run identity")
    source_config = json.loads(configuration.read_text(encoding="utf-8"))
    source_run = json.loads(identity_file.read_text(encoding="utf-8"))
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if source_run.get("model_id") != MODEL_ID or source_run.get("model_revision") != MODEL_REVISION:
        raise ValueError("Warm-start source does not originate from the required Qwen3 base")
    if source_config.get("model_type") != "qwen3" or source_config.get("quantization_config"):
        raise ValueError("Warm-start source must be an unquantized Qwen3 token classifier")
    if (source_config.get("privacy_attention_mode", "causal") != attention_mode
            or source_run.get("attention_mode") != attention_mode):
        raise ValueError("Warm-start source attention mode does not match the new experiment")
    source_labels = source_config.get("id2label", {})
    if len(source_labels) != len(names) or any(source_labels.get(str(i)) != name for i, name in enumerate(names)):
        raise ValueError("Warm-start source label order does not match the target dataset")
    if any(source_config.get(field) != getattr(target_config, field, None) for field in ARCHITECTURE_FIELDS):
        raise ValueError("Warm-start source architecture does not match the target model")
    step = state.get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("Warm-start checkpoint has invalid Trainer state")
    return {
        "kind": "checkpoint_weights_only", "weights_sha256": _sha256_file(weights),
        "config_sha256": _sha256_file(configuration), "source_checkpoint": checkpoint.name,
        "source_step": step, "source_epoch": state.get("epoch"), "source_model_id": MODEL_ID,
        "source_model_revision": MODEL_REVISION, "source_attention_mode": attention_mode,
        "source_dataset_manifest_sha256": source_run.get("dataset_manifest_sha256"),
        "source_protocol_sha256": source_config.get("privacy_continuation_protocol_sha256") or source_run.get("protocol_sha256"),
    }


def load_initial_model(checkpoint, target_config):
    model, loading = PrivacyFilterModel.from_pretrained(
        checkpoint, config=target_config, torch_dtype=torch.float32, use_safetensors=True,
        attn_implementation="sdpa", output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Warm-start checkpoint weights do not exactly match the classifier architecture")
    state = model.state_dict()
    last_layer = model.config.num_hidden_layers - 1
    checked = ("score.weight", "score.bias", "model.layers.0.self_attn.q_proj.weight",
               f"model.layers.{last_layer}.self_attn.q_proj.weight")
    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt", device="cpu") as stored:
        for key in checked:
            if key not in stored.keys() or key not in state or not torch.equal(stored.get_tensor(key), state[key].detach().cpu()):
                raise ValueError("Warm-start checkpoint weights failed exact verification")
    return model


def run_identity(config, dataset_manifest_sha256, initialization=None):
    mode = config.get("attention_mode", "causal")
    protocol = {key: value for key, value in config.items() if key not in {"dataset_dir", "output_dir", "attention_mode"}}
    base_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = {"attention_mode": mode, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "dataset_manifest_sha256": dataset_manifest_sha256, "seed": int(config.get("seed", 42)),
                "protocol_sha256": base_sha}
    if initialization is not None:
        effective = {"base_protocol_sha256": base_sha, "initialization": initialization}
        identity["base_protocol_sha256"] = base_sha
        identity["protocol_sha256"] = hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        identity["initialization"] = initialization
    return identity


def select_nvidia_batch(dataset_dir, minimum_tokens=1):
    """Pick one supervised NVIDIA training window without exposing its text."""
    import pyarrow.dataset as ds

    if type(minimum_tokens) is not int or minimum_tokens < 1:
        raise ValueError("minimum_tokens must be a positive integer")
    source = ds.dataset(Path(dataset_dir) / "train.parquet", format="parquet")
    scanner = source.scanner(columns=["input_ids", "attention_mask", "labels"],
                             filter=ds.field("source") == "nemotron", batch_size=64)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if (len(row["input_ids"]) >= minimum_tokens and any(label > 0 for label in row["labels"])
                    and len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])):
                return {key: torch.tensor([row[key]], dtype=torch.long) for key in ("input_ids", "attention_mask", "labels")}
    raise ValueError("No supervised NVIDIA training window found")
