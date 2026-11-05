import torch

from valendra_privacy_filter.training.smoke import train_step, make_smoke_batch
from valendra_privacy_filter.training.train import (token_span_metrics, validate_resume_checkpoint,
                                                validation_subset, prioritize_spanish_training,
                                                WarmupAwareEarlyStopping, MinimumEpochEarlyStopping, build_early_stopping,
                                                continuation_policy)
from valendra_privacy_filter.labels import label_names
from test_model import tiny_model


def test_bf16_autocast_updates_backbone_and_uses_fp32_adam_states():
    model = tiny_model().float()
    batch = {
        "input_ids": torch.tensor([[2, 3, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "labels": torch.tensor([[0, 1, 2, -100]]),
    }
    report = train_step(model, batch, device="cpu")
    assert report["parameter_dtype"] == "torch.float32"
    assert report["logits_dtype"] == "torch.bfloat16"
    assert report["gradient_dtype"] == "torch.float32"
    assert report["optimizer_state_dtype"] == "torch.float32"
    assert report["head_updated"] and report["backbone_updated"]
    assert report["loss"] > 0


def test_smoke_batch_contains_real_bioes_entities_at_requested_length():
    class CharacterTokenizer:
        def __call__(self, text, max_length, **kwargs):
            text = text[:max_length]
            return {"input_ids": torch.tensor([[ord(c) for c in text]]),
                    "attention_mask": torch.ones((1, len(text)), dtype=torch.long),
                    "offset_mapping": torch.tensor([[(i, i + 1) for i in range(len(text))]])}

    batch = make_smoke_batch(CharacterTokenizer(), sequence_length=64)
    assert batch["input_ids"].shape == (1, 64)
    assert batch["labels"].shape == (1, 64)
    assert (batch["labels"] > 0).any()


def test_training_metric_ignores_padding_and_requires_exact_bioes_span():
    names = label_names()
    gold = torch.tensor([[0, names.index("B-PERSON"), names.index("E-PERSON"), -100]])
    logits = torch.full((1, 4, len(names)), -10.0)
    for position, tag in enumerate(["O", "B-PERSON", "E-PERSON", "S-EMAIL"]):
        logits[0, position, names.index(tag)] = 10.0
    result = token_span_metrics(logits.numpy(), gold.numpy(), names)
    assert result["f1"] == 1.0
    assert result["recall"] == 1.0


def test_resume_requires_checkpoint_from_matching_run(tmp_path):
    import json
    import pytest

    run = tmp_path / "run"
    checkpoint = run / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    identity = {"attention_mode": "causal", "dataset": "sha256"}
    (run / "run_identity.json").write_text(json.dumps(identity))
    (checkpoint / "config.json").write_text(json.dumps({"privacy_attention_mode": "causal"}))
    validate_resume_checkpoint(checkpoint, run, identity)
    with pytest.raises(ValueError, match="run"):
        validate_resume_checkpoint(checkpoint, tmp_path / "other", identity)
    with pytest.raises(ValueError, match="identity"):
        validate_resume_checkpoint(checkpoint, run, {"attention_mode": "causal", "dataset": "different"})


def test_validation_subset_is_seeded_and_does_not_change_training():
    from datasets import Dataset

    dataset = Dataset.from_dict({"id": list(range(40))})
    first = validation_subset(dataset, 8, 42)
    assert len(first) == 8
    assert first["id"] == validation_subset(dataset, 8, 42)["id"]
    assert len(dataset) == 40


def test_training_prioritizes_spanish_spain_without_dropping_other_examples():
    from datasets import Dataset

    dataset = Dataset.from_dict({"id": [0, 1, 2], "language": ["es", "es", "en"], "region": ["ES", "MX", "US"]})
    prioritized = prioritize_spanish_training(dataset, repeats=3)
    assert list(prioritized["id"]).count(0) == 3
    assert list(prioritized["id"]).count(1) == 1
    assert list(prioritized["id"]).count(2) == 1
    assert len(dataset) == 3


def test_early_stopping_does_not_expire_during_warmup():
    from types import SimpleNamespace
    from transformers import TrainerControl

    args = SimpleNamespace(metric_for_best_model="f1", greater_is_better=True, get_warmup_steps=lambda _: 10)
    state = SimpleNamespace(max_steps=100, global_step=5, best_metric=0.5)
    control = TrainerControl()
    callback = WarmupAwareEarlyStopping(early_stopping_patience=2)
    for _ in range(3):
        callback.on_evaluate(args, state, control, {"eval_f1": 0.0})
    assert not control.should_training_stop
    state.global_step = 11
    callback.on_evaluate(args, state, control, {"eval_f1": 0.0})
    callback.on_evaluate(args, state, control, {"eval_f1": 0.0})
    assert control.should_training_stop


def test_fresh_experiment_can_require_one_epoch_before_stopping():
    import pytest

    callback = build_early_stopping({"epochs": 3, "min_early_stopping_epoch": 1.0}, None)
    assert isinstance(callback, MinimumEpochEarlyStopping)
    assert callback.minimum_epoch == 1.0
    with pytest.raises(ValueError, match="epoch"):
        build_early_stopping({"epochs": 3, "min_early_stopping_epoch": 4.0}, None)
    with pytest.raises(ValueError, match="epoch"):
        build_early_stopping({"epochs": 3, "min_early_stopping_epoch": 2.0}, {"minimum_epoch": 1.0})


def test_mixed_attention_configs_use_same_data_and_training_protocol():
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[1] / "configs"
    causal = yaml.safe_load((root / "train-openpii-nemotron-causal.yaml").read_text())
    bidirectional = yaml.safe_load((root / "train-openpii-nemotron-bidirectional.yaml").read_text())
    assert causal["attention_mode"] == "causal"
    assert bidirectional["attention_mode"] == "bidirectional"
    assert causal["dataset_dir"] == bidirectional["dataset_dir"] == "data/processed-openpii-nemotron-v1"
    assert causal["output_dir"] != bidirectional["output_dir"]
    assert causal["min_early_stopping_epoch"] == bidirectional["min_early_stopping_epoch"] == 1.0
    excluded = {"attention_mode", "output_dir"}
    assert {k: v for k, v in causal.items() if k not in excluded} == {k: v for k, v in bidirectional.items() if k not in excluded}


def test_minimum_epoch_guard_is_persisted_and_defers_stopping():
    from types import SimpleNamespace
    from transformers import TrainerControl

    args = SimpleNamespace(metric_for_best_model="f1", greater_is_better=True, get_warmup_steps=lambda _: 10)
    state = SimpleNamespace(max_steps=100, global_step=50, epoch=0.99, best_metric=0.8)
    control = TrainerControl()
    callback = MinimumEpochEarlyStopping(early_stopping_patience=2, minimum_epoch=1.0)
    for _ in range(3):
        callback.on_evaluate(args, state, control, {"eval_f1": 0.5})
    assert callback.early_stopping_patience_counter == 0
    assert not control.should_training_stop
    assert callback.state()["args"]["minimum_epoch"] == 1.0
    state.epoch = 1.0
    callback.on_evaluate(args, state, control, {"eval_f1": 0.5})
    callback.on_evaluate(args, state, control, {"eval_f1": 0.5})
    assert control.should_training_stop


def test_continuation_policy_binds_checkpoint_and_survives_restart(tmp_path):
    import json
    import pytest

    run = tmp_path / "run"
    checkpoint = run / "checkpoint-8000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 8000, "epoch": 0.195}))
    identity = {"protocol_sha256": "base-protocol", "dataset_manifest_sha256": "dataset"}
    policy = continuation_policy(run, identity, checkpoint, minimum_epoch=1.0)
    assert policy["source_step"] == 8000
    assert policy["minimum_epoch"] == 1.0
    assert policy["protocol_sha256"] != identity["protocol_sha256"]
    assert continuation_policy(run, identity, checkpoint, minimum_epoch=None) == policy
    with pytest.raises(ValueError, match="policy"):
        continuation_policy(run, identity, checkpoint, minimum_epoch=2.0)
    older = run / "checkpoint-6500"
    older.mkdir()
    (older / "trainer_state.json").write_text(json.dumps({"global_step": 6500, "epoch": 0.16}))
    with pytest.raises(ValueError, match="earlier"):
        continuation_policy(run, identity, older, minimum_epoch=None)


def test_real_trainer_resume_after_early_stop_reaches_minimum_epoch(tmp_path):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = Dataset.from_dict({"input_ids": [[2, 3, 4, 5]] * 16,
                                     "attention_mask": [[1, 1, 1, 1]] * 16,
                                     "labels": [[0, 1, 3, 0]] * 16})
        args = TrainingArguments(output_dir=str(tmp_path), num_train_epochs=2, per_device_train_batch_size=2,
                                 per_device_eval_batch_size=4, eval_strategy="steps", save_strategy="steps",
                                 eval_steps=2, save_steps=2, logging_steps=2, learning_rate=1e-3, use_cpu=True,
                                 save_safetensors=True, load_best_model_at_end=True, metric_for_best_model="f1",
                                 greater_is_better=True, report_to=[], disable_tqdm=True,
                                 restore_callback_states_from_checkpoint=True)

        def make_trainer(callback):
            set_seed(42)
            return Trainer(model=tiny_model(), args=args, train_dataset=dataset, eval_dataset=dataset,
                           data_collator=default_data_collator, compute_metrics=lambda _: {"f1": 0.0},
                           callbacks=[callback])

        first = make_trainer(WarmupAwareEarlyStopping(early_stopping_patience=1))
        first.train()
        assert first.state.global_step == 4
        checkpoint = tmp_path / "checkpoint-4"
        assert (checkpoint / "optimizer.pt").is_file()
        resumed = make_trainer(MinimumEpochEarlyStopping(early_stopping_patience=1, minimum_epoch=1.0))
        resumed.train(resume_from_checkpoint=str(checkpoint))
        assert resumed.state.global_step == 8
        assert resumed.state.epoch >= 1.0
        state = resumed.state.stateful_callbacks["MinimumEpochEarlyStopping"]
        assert state["args"]["minimum_epoch"] == 1.0
        restarted = make_trainer(MinimumEpochEarlyStopping(early_stopping_patience=1, minimum_epoch=2.0))
        restarted.train(resume_from_checkpoint=str(tmp_path / "checkpoint-6"))
        assert restarted.state.global_step == 8
        callback = next(cb for cb in restarted.callback_handler.callbacks if isinstance(cb, MinimumEpochEarlyStopping))
        assert callback.minimum_epoch == 1.0
    finally:
        torch.set_num_threads(previous_threads)
