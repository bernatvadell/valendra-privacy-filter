"""BF16 forward/backward/update and official-checkpoint smoke test."""

from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from valendra_privacy_filter.datasets.alignment import align_labels
from valendra_privacy_filter.labels import label_names
from valendra_privacy_filter.models import PrivacyFilterModel

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def make_smoke_batch(tokenizer, sequence_length=512):
    if not 2 <= sequence_length <= 32768:
        raise ValueError("sequence_length must be between 2 and 32768")
    unit = "Contact Ana García at ana@example.com. "
    text = unit * sequence_length
    spans = []
    for number in range(sequence_length):
        for value, category in (("Ana García", "PERSON"), ("ana@example.com", "EMAIL")):
            start = number * len(unit) + unit.index(value)
            spans.append({"start": start, "end": start + len(value), "label": category})
    batch = tokenizer(text, max_length=sequence_length, truncation=True, padding="max_length",
                      return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = batch.pop("offset_mapping")[0].tolist()
    batch["labels"] = torch.tensor([align_labels(text, offsets, spans, [], label_names())])
    return batch


def train_step(model, batch, device="cuda"):
    device = torch.device(device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Smoke supports CUDA or explicitly selected CPU")
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16-capable CUDA is required")
    model = model.to(device=device, dtype=torch.float32).train()
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Full fine-tuning requires all parameters trainable")
    batch = {key: value.to(device) for key, value in batch.items()}
    backbone = model.model.layers[0].self_attn.q_proj.weight
    head = model.score.weight
    before_backbone = backbone.detach().reshape(-1)[:4096].clone()
    before_head = head.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(**batch)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("Nonfinite or absent training loss")
    output.loss.backward()
    if backbone.grad is None or head.grad is None:
        raise RuntimeError("Backbone or head has no gradient")
    gradient_dtype = str(backbone.grad.dtype)
    optimizer.step()
    state_dtype = str(optimizer.state[backbone]["exp_avg"].dtype)
    report = {
        "loss": float(output.loss.detach()), "parameter_dtype": str(backbone.dtype),
        "logits_dtype": str(output.logits.dtype), "gradient_dtype": gradient_dtype,
        "optimizer_state_dtype": state_dtype,
        "backbone_updated": not torch.equal(before_backbone, backbone.detach().reshape(-1)[:4096]),
        "head_updated": not torch.equal(before_head, head.detach()),
    }
    if not report["head_updated"] or not report["backbone_updated"]:
        raise RuntimeError("Optimizer did not update both head and backbone")
    return report


def official_smoke(output_dir, device="cuda", attention_mode="causal", sequence_length=512, gradient_checkpointing=True):
    started = time.perf_counter()
    names = label_names()
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, num_labels=len(names))
    config.id2label = dict(enumerate(names))
    config.label2id = {name: i for i, name in enumerate(names)}
    config.privacy_attention_mode = attention_mode
    model, loading = PrivacyFilterModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, config=config, torch_dtype=torch.float32,
        attn_implementation="sdpa", use_safetensors=True, output_loading_info=True,
    )
    unexpected_missing = set(loading["missing_keys"]) - {"score.weight", "score.bias"}
    if unexpected_missing or loading["mismatched_keys"]:
        raise RuntimeError(f"Unexpected checkpoint incompatibility: {unexpected_missing}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    batch = make_smoke_batch(tokenizer, sequence_length)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    report = train_step(model, batch, device=device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    if not (output_dir / "model.safetensors").is_file():
        raise RuntimeError("Safetensors checkpoint missing")
    expected_head = model.score.weight.detach().cpu().clone()
    del model
    if device.startswith("cuda"):
        report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
    reloaded = PrivacyFilterModel.from_pretrained(output_dir, torch_dtype=torch.float32)
    torch.testing.assert_close(expected_head, reloaded.score.weight.detach().cpu(), rtol=0, atol=0)
    report.update({"checkpoint_reloaded": True, "missing_keys": loading["missing_keys"],
                   "unexpected_keys": loading["unexpected_keys"], "seconds": time.perf_counter() - started,
                   "model_id": MODEL_ID, "revision": MODEL_REVISION, "mode": attention_mode,
                   "sequence_length": int(batch["input_ids"].shape[1]),
                   "positive_tokens": int((batch["labels"] > 0).sum()),
                   "gradient_checkpointing": gradient_checkpointing})
    return report
