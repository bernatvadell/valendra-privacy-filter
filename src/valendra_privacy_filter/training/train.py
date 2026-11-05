"""Reproducible full-parameter BF16 fine-tuning with Hugging Face Trainer."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoConfig, AutoTokenizer, DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from valendra_privacy_filter.evaluation.metrics import _metrics
from valendra_privacy_filter.datasets.prepare import verify_dataset
from valendra_privacy_filter.models import PrivacyFilterModel
from valendra_privacy_filter.models.decode import viterbi_decode
from valendra_privacy_filter.training.initialization import checkpoint_provenance, load_initial_model, run_identity
from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION


class WarmupAwareEarlyStopping(EarlyStoppingCallback):
    """Do not consume early-stopping patience before warmup has finished."""

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.global_step <= args.get_warmup_steps(state.max_steps):
            self.early_stopping_patience_counter = 0
            return
        return super().on_evaluate(args, state, control, metrics, **kwargs)


class MinimumEpochEarlyStopping(WarmupAwareEarlyStopping):
    """Allow early stopping only after both warmup and a full minimum epoch."""

    def __init__(self, early_stopping_patience=1, early_stopping_threshold=0.0, minimum_epoch=1.0):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        if not math.isfinite(minimum_epoch) or minimum_epoch <= 0:
            raise ValueError("minimum_epoch must be positive and finite")
        self.minimum_epoch = float(minimum_epoch)

    def on_train_begin(self, args, state, control, **kwargs):
        # A resumed Trainer retains the old callback-state dictionary; register
        # this new callback before Transformers saves the next checkpoint.
        state.stateful_callbacks.setdefault(self.__class__.__name__, self.state())
        return super().on_train_begin(args, state, control, **kwargs)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch is None or state.epoch < self.minimum_epoch:
            self.early_stopping_patience_counter = 0
            return
        return super().on_evaluate(args, state, control, metrics, **kwargs)

    def state(self):
        result = super().state()
        result["args"]["minimum_epoch"] = self.minimum_epoch
        return result


def continuation_policy(output_dir, identity, checkpoint, minimum_epoch=None):
    """Persist an explicit protocol revision without changing the parent run identity."""
    path = Path(output_dir) / "continuation_policy.json"
    if path.exists():
        policy = json.loads(path.read_text(encoding="utf-8"))
        if (policy["base_protocol_sha256"] != identity["protocol_sha256"]
                or policy["dataset_manifest_sha256"] != identity["dataset_manifest_sha256"]
                or (minimum_epoch is not None and float(minimum_epoch) != policy["minimum_epoch"])):
            raise ValueError("Existing continuation policy does not match the requested run")
        if checkpoint is not None:
            checkpoint_state = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
            if checkpoint_state["global_step"] < policy["source_step"]:
                raise ValueError("Cannot apply continuation policy to an earlier checkpoint")
        return policy
    if minimum_epoch is None:
        return None
    minimum_epoch = float(minimum_epoch)
    if not math.isfinite(minimum_epoch) or minimum_epoch <= 0:
        raise ValueError("minimum_epoch must be positive and finite")
    if checkpoint is None:
        raise ValueError("A checkpoint is required for a continuation policy")
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    settings = {"base_protocol_sha256": identity["protocol_sha256"], "minimum_epoch": minimum_epoch,
                "policy": "early_stopping_after_minimum_epoch_v1"}
    protocol_sha = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    policy = {**settings, "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
              "protocol_sha256": protocol_sha, "source_checkpoint": checkpoint.name,
              "source_step": state["global_step"], "source_epoch": state["epoch"],
              "created_utc": datetime.now(timezone.utc).isoformat()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(policy, stream, indent=2)
    return policy


def build_early_stopping(config, policy):
    patience = int(config.get("early_stopping_patience", 3))
    if patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    configured = config.get("min_early_stopping_epoch")
    if configured is not None:
        configured = float(configured)
    minimum = float(policy["minimum_epoch"]) if policy else configured
    if policy and configured is not None and minimum < configured:
        raise ValueError("Continuation minimum epoch cannot weaken the configured minimum epoch")
    if minimum is None:
        return WarmupAwareEarlyStopping(early_stopping_patience=patience)
    if not math.isfinite(minimum) or minimum <= 0 or minimum > float(config.get("epochs", 3)):
        raise ValueError("Minimum stopping epoch must be positive and not exceed configured epochs")
    return MinimumEpochEarlyStopping(early_stopping_patience=patience, minimum_epoch=minimum)


def _spans(tags, names):
    result, active = set(), None
    for index, tag in enumerate(tags):
        name = names[int(tag)]
        if name == "O":
            active = None
            continue
        prefix, category = name.split("-", 1)
        if prefix == "S":
            result.add((index, index + 1, category))
            active = None
        elif prefix == "B":
            active = (index, category)
        elif prefix == "E" and active and active[1] == category:
            result.add((active[0], index + 1, category))
            active = None
        elif prefix != "I" or not active or active[1] != category:
            active = None
    return result


def token_span_metrics(logits, labels, names):
    """Exact spans on supervised contiguous token regions, used for early stopping."""
    tp = fp = fn = 0
    for emissions, gold in zip(logits, labels):
        valid = np.asarray(gold) != -100
        start = 0
        while start < len(gold):
            if not valid[start]:
                start += 1
                continue
            end = start + 1
            while end < len(gold) and valid[end]:
                end += 1
            predicted = viterbi_decode(torch.as_tensor(emissions[start:end]), names)
            predicted_spans = _spans(predicted, names)
            gold_spans = _spans(gold[start:end], names)
            tp += len(predicted_spans & gold_spans)
            fp += len(predicted_spans - gold_spans)
            fn += len(gold_spans - predicted_spans)
            start = end
    return _metrics(tp, fp, fn)


def validate_resume_checkpoint(checkpoint, output_dir, identity):
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    if checkpoint.parent != output_dir:
        raise ValueError("Resume checkpoint must belong to the same run directory")
    identity_path = output_dir / "run_identity.json"
    if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
        raise ValueError("Resume run identity does not match the requested experiment")
    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    if checkpoint_config.get("privacy_attention_mode", "causal") != identity["attention_mode"]:
        raise ValueError("Checkpoint attention mode differs from experiment")
    initialization = identity.get("initialization")
    if initialization and checkpoint_config.get("privacy_initialization_weights_sha256") != initialization["weights_sha256"]:
        raise ValueError("Checkpoint initialization does not match the recorded run")


def validation_subset(dataset, limit, seed):
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("max_eval_samples must be positive")
    return dataset.shuffle(seed=seed).select(range(min(len(dataset), limit)))


def prioritize_spanish_training(dataset, repeats=1):
    if repeats < 1:
        raise ValueError("spanish_train_repeats must be at least 1")
    if repeats == 1:
        return dataset
    spanish = [index for index, (language, region) in enumerate(zip(dataset["language"], dataset["region"]))
               if language == "es" and region == "ES"]
    return dataset.select(list(range(len(dataset))) + spanish * (repeats - 1))


def validate_training_manifest(manifest, require_complete_nvidia=False):
    if manifest.get("group_cross_split_count") != 0:
        raise ValueError("Dataset group leakage check is absent or failed")
    if require_complete_nvidia:
        source = manifest.get("source_parameters", {})
        source_ids = {item.get("id") for item in manifest.get("sources", [])}
        counts = manifest.get("splits", {})
        if ("nvidia/Nemotron-PII" not in source_ids or not source.get("nvidia_train_is_complete")
                or source.get("sampled_nvidia_holdout", True)
                or not manifest.get("postwrite_isolation_audit", {}).get("passed")
                or not counts.get("train", {}).get("sources", {}).get("nemotron")
                or not counts.get("validation", {}).get("sources", {}).get("nemotron")):
            raise ValueError("Full NVIDIA data and a passing holdout isolation audit are required")


def validate_warmstart_lineage(initialization, manifest):
    baseline = manifest.get("source_parameters", {}).get("baseline_manifest_sha256")
    if (not baseline or baseline != initialization.get("source_dataset_manifest_sha256")
            or not manifest.get("postwrite_isolation_audit", {}).get("passed")):
        raise ValueError("Target holdouts were not verified against the warm-start source dataset")


def train(config, resume_from_checkpoint=None, min_early_stopping_epoch=None, initialize_from_checkpoint=None):
    if resume_from_checkpoint and initialize_from_checkpoint:
        raise ValueError("Warm-start initialization and same-phase resume are mutually exclusive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Full training requires a CUDA device supporting BF16")
    mode = config.get("attention_mode", "causal")
    if mode not in {"causal", "bidirectional"}:
        raise ValueError("Unknown attention mode")
    dataset_dir = Path(config["dataset_dir"])
    manifest_path = dataset_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = verify_dataset(dataset_dir, ["train", "validation"])
    names = manifest["label_names"]
    validate_training_manifest(manifest, require_complete_nvidia=bool(config.get("require_complete_nvidia", False)))
    model_config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, num_labels=len(names))
    model_config.id2label = dict(enumerate(names))
    model_config.label2id = {name: i for i, name in enumerate(names)}
    model_config.privacy_attention_mode = mode
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run_identity.json"
    existing_identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
    if initialize_from_checkpoint and existing_identity:
        raise ValueError("Existing experiment: use a new output directory for warm-start training")
    if initialize_from_checkpoint:
        initialization = checkpoint_provenance(initialize_from_checkpoint, model_config, names, mode)
    else:
        initialization = existing_identity.get("initialization") if existing_identity and resume_from_checkpoint else None
    if initialization:
        validate_warmstart_lineage(initialization, manifest)
    identity = run_identity(config, hashlib.sha256(manifest_bytes).hexdigest(), initialization)
    if existing_identity and existing_identity != identity:
        raise ValueError("Output directory belongs to a different experiment")
    if resume_from_checkpoint:
        validate_resume_checkpoint(resume_from_checkpoint, output_dir, identity)
    elif identity_path.exists():
        raise ValueError("Existing experiment: explicitly resume or use a fresh output directory")
    elif min_early_stopping_epoch is not None:
        raise ValueError("A checkpoint is required for a minimum-epoch continuation")
    if min_early_stopping_epoch is not None and float(min_early_stopping_epoch) > float(config.get("epochs", 3)):
        raise ValueError("Minimum stopping epoch cannot exceed configured epochs")
    policy = continuation_policy(output_dir, identity, resume_from_checkpoint, min_early_stopping_epoch)
    if policy is not None and policy["minimum_epoch"] > float(config.get("epochs", 3)):
        raise ValueError("Minimum stopping epoch cannot exceed configured epochs")
    early_stopping = build_early_stopping(config, policy)
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    set_seed(identity["seed"])
    data = load_dataset("parquet", data_files={"train": str(dataset_dir / "train.parquet"),
                                               "validation": str(dataset_dir / "validation.parquet")})
    data = data.filter(lambda row: any(label != -100 for label in row["labels"]))
    if not len(data["train"]) or not len(data["validation"]):
        raise ValueError("Training and validation need supervised examples")
    eval_limit = config.get("max_eval_samples")
    evaluation_data = validation_subset(data["validation"], int(eval_limit) if eval_limit is not None else None,
                                        identity["seed"])
    training_data = prioritize_spanish_training(data["train"], int(config.get("spanish_train_repeats", 1)))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    if policy:
        model_config.privacy_continuation_protocol_sha256 = policy["protocol_sha256"]
        model_config.privacy_minimum_stopping_epoch = policy["minimum_epoch"]
    elif config.get("min_early_stopping_epoch") is not None:
        model_config.privacy_minimum_stopping_epoch = float(config["min_early_stopping_epoch"])
    if initialization:
        model_config.privacy_initialization_weights_sha256 = initialization["weights_sha256"]
    if initialize_from_checkpoint:
        model = load_initial_model(initialize_from_checkpoint, model_config)
    else:
        model = PrivacyFilterModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, config=model_config,
                                                    torch_dtype=torch.float32, use_safetensors=True,
                                                    attn_implementation="sdpa")
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise RuntimeError("Expected unfrozen FP32 master parameters")
    save_steps = int(config.get("save_steps", 500))
    args = TrainingArguments(
        output_dir=str(output_dir), bf16=True, fp16=False, seed=identity["seed"], data_seed=identity["seed"],
        per_device_train_batch_size=int(config.get("train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(config.get("epochs", 3)), max_steps=int(config.get("max_steps", -1)),
        learning_rate=float(config.get("learning_rate", 2e-5)), weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)), lr_scheduler_type=config.get("scheduler", "linear"),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)), optim="adamw_torch",
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", save_strategy="steps", eval_steps=save_steps, save_steps=save_steps,
        logging_steps=int(config.get("logging_steps", 25)), save_total_limit=int(config.get("save_total_limit", 2)),
        save_safetensors=True, load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        restore_callback_states_from_checkpoint=True,
        eval_accumulation_steps=int(config.get("eval_accumulation_steps", 16)), report_to=[],
        disable_tqdm=True, dataloader_num_workers=int(config.get("workers", 0)),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=training_data, eval_dataset=evaluation_data,
        processing_class=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100),
        compute_metrics=lambda prediction: token_span_metrics(prediction.predictions, prediction.label_ids, names),
        callbacks=[early_stopping],
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    return {"train": result.metrics, "validation": trainer.evaluate(), "validation_windows": len(evaluation_data),
            "unique_training_windows": len(data["train"]), "effective_training_windows": len(training_data),
            "final_checkpoint": str(final_dir), "continuation_policy": policy, **identity}
