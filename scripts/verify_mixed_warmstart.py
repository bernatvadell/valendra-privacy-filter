"""Run one disposable BF16 update from prior weights on a real NVIDIA training window."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from valendra_privacy_filter.datasets.prepare import verify_dataset
from valendra_privacy_filter.training.initialization import checkpoint_provenance, load_initial_model, select_nvidia_batch
from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION, train_step
from valendra_privacy_filter.training.train import validate_training_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--min-tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA is required")
    data = Path(args.dataset_dir)
    manifest = verify_dataset(data, ["train", "validation"])
    validate_training_manifest(manifest, require_complete_nvidia=True)
    names = manifest["label_names"]
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, num_labels=len(names))
    config.id2label = dict(enumerate(names))
    config.label2id = {name: i for i, name in enumerate(names)}
    config.privacy_attention_mode = "causal"
    provenance = checkpoint_provenance(args.checkpoint, config, names, "causal")
    model = load_initial_model(args.checkpoint, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batch = select_nvidia_batch(data, minimum_tokens=args.min_tokens)
    torch.cuda.reset_peak_memory_stats()
    result = train_step(model, batch, device="cuda")
    result.update({"source_step": provenance["source_step"], "source_weights_sha256": provenance["weights_sha256"],
                   "dataset_manifest_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
                   "input_source": "nemotron", "sequence_tokens": int(batch["input_ids"].shape[1]),
                   "positive_labels": int((batch["labels"] > 0).sum()),
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(), "optimizer": "fresh_adamw"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
