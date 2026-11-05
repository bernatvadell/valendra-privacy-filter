import copy
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from valendra_privacy_filter.labels import label_names
from valendra_privacy_filter.models import PrivacyFilterModel
from valendra_privacy_filter.training.initialization import (checkpoint_provenance, load_initial_model,
                                                             run_identity, select_nvidia_batch)
from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION
from valendra_privacy_filter.training.train import (train, validate_resume_checkpoint, validate_training_manifest,
                                                   validate_warmstart_lineage)
from test_model import tiny_model


@pytest.fixture
def source_checkpoint(tmp_path):
    source = tmp_path / "old-run" / "checkpoint-13500"
    source.mkdir(parents=True)
    model = tiny_model().eval()
    names = label_names()
    model.config.id2label = dict(enumerate(names))
    model.config.label2id = {name: i for i, name in enumerate(names)}
    model.config.privacy_continuation_protocol_sha256 = "old-continuation-protocol"
    model.save_pretrained(source, safe_serialization=True)
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 13500, "epoch": 0.3297}))
    (source.parent / "run_identity.json").write_text(json.dumps({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "attention_mode": "causal",
        "dataset_manifest_sha256": "old-dataset", "protocol_sha256": "old-protocol", "seed": 42,
    }))
    return source, model, names


def test_warmstart_loads_both_trained_backbone_and_classifier(source_checkpoint):
    source, trained, names = source_checkpoint
    target_config = copy.deepcopy(trained.config)
    info = checkpoint_provenance(source, target_config, names, "causal")
    initialized = load_initial_model(source, target_config)
    torch.testing.assert_close(initialized.score.weight, trained.score.weight, rtol=0, atol=0)
    torch.testing.assert_close(initialized.model.layers[0].self_attn.q_proj.weight,
                               trained.model.layers[0].self_attn.q_proj.weight, rtol=0, atol=0)
    assert info["source_step"] == 13500
    assert info["source_protocol_sha256"] == "old-continuation-protocol"
    assert info["weights_sha256"] == hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()


def test_warmstart_rejects_incompatible_labels_architecture_and_attention(source_checkpoint):
    source, trained, names = source_checkpoint
    different = list(names)
    different[1], different[2] = different[2], different[1]
    with pytest.raises(ValueError, match="label"):
        checkpoint_provenance(source, trained.config, different, "causal")
    wrong_architecture = copy.deepcopy(trained.config)
    wrong_architecture.hidden_size += 1
    with pytest.raises(ValueError, match="architecture"):
        checkpoint_provenance(source, wrong_architecture, names, "causal")
    with pytest.raises(ValueError, match="attention"):
        checkpoint_provenance(source, trained.config, names, "bidirectional")


def test_warmstart_rejects_wrong_source_and_missing_classifier_weights(source_checkpoint):
    source, trained, names = source_checkpoint
    identity_path = source.parent / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["model_id"] = "not-the-required-base"
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="base"):
        checkpoint_provenance(source, trained.config, names, "causal")
    weights = load_file(source / "model.safetensors")
    weights.pop("score.weight")
    save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="weights"):
        load_initial_model(source, copy.deepcopy(trained.config))


def test_warmstart_is_a_distinct_protocol_and_can_resume_without_source(tmp_path, source_checkpoint):
    import shutil

    source, trained, names = source_checkpoint
    info = checkpoint_provenance(source, trained.config, names, "causal")
    config = {"attention_mode": "causal", "seed": 42, "epochs": 3}
    cold = run_identity(config, "new-dataset")
    warm = run_identity(config, "new-dataset", initialization=info)
    assert warm["protocol_sha256"] != cold["protocol_sha256"]
    assert warm["base_protocol_sha256"] == cold["protocol_sha256"]
    assert run_identity(config, "new-dataset", initialization=warm["initialization"]) == warm
    run, checkpoint = tmp_path / "new-run", tmp_path / "new-run/checkpoint-1"
    checkpoint.mkdir(parents=True)
    (run / "run_identity.json").write_text(json.dumps(warm))
    (checkpoint / "config.json").write_text(json.dumps({"privacy_attention_mode": "causal",
        "privacy_initialization_weights_sha256": info["weights_sha256"]}))
    shutil.rmtree(source)
    validate_resume_checkpoint(checkpoint, run, warm)
    (checkpoint / "config.json").write_text(json.dumps({"privacy_attention_mode": "causal",
        "privacy_initialization_weights_sha256": "wrong"}))
    with pytest.raises(ValueError, match="initial"):
        validate_resume_checkpoint(checkpoint, run, warm)


def test_warmstarted_trainer_starts_at_zero_with_fresh_adamw_state(tmp_path, source_checkpoint):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, default_data_collator

    source, trained, _ = source_checkpoint
    initialized = load_initial_model(source, copy.deepcopy(trained.config))
    before = initialized.score.weight.detach().clone()
    dataset = Dataset.from_dict({"input_ids": [[2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1]],
                                 "labels": [[0, 1, 3, 0]]})
    args = TrainingArguments(output_dir=str(tmp_path / "new-phase"), max_steps=1, use_cpu=True,
                             learning_rate=1e-3, optim="adamw_torch", save_strategy="no", report_to=[],
                             disable_tqdm=True)
    trainer = Trainer(model=initialized, args=args, train_dataset=dataset, data_collator=default_data_collator)
    trainer.create_optimizer_and_scheduler(num_training_steps=1)
    assert trainer.state.global_step == 0
    assert len(trainer.optimizer.state) == 0
    trainer.train()
    assert trainer.state.global_step == 1
    assert len(trainer.optimizer.state) > 0
    assert not torch.equal(before, trainer.model.score.weight.detach())


def test_warmstarted_phase_resumes_exactly_without_original_source(tmp_path, source_checkpoint):
    import shutil
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

    source, trained, names = source_checkpoint
    info = checkpoint_provenance(source, trained.config, names, "causal")
    target = copy.deepcopy(trained.config)
    target._attn_implementation = "sdpa"
    target.privacy_initialization_weights_sha256 = info["weights_sha256"]
    initial = load_initial_model(source, target)
    dataset = Dataset.from_dict({"input_ids": [[2, 3, 4, 5]] * 4,
                                 "attention_mask": [[1, 1, 1, 1]] * 4,
                                 "labels": [[0, 1, 3, 0]] * 4})
    args = TrainingArguments(output_dir=str(tmp_path / "warm-run"), max_steps=2, use_cpu=True,
                             per_device_train_batch_size=1, learning_rate=1e-3, optim="adamw_torch",
                             save_strategy="steps", save_steps=1, report_to=[], disable_tqdm=True, seed=42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        set_seed(42)
        original = Trainer(model=initial, args=args, train_dataset=dataset, data_collator=default_data_collator)
        original.train()
        expected = original.model.score.weight.detach().clone()
        first_step = tmp_path / "warm-run/checkpoint-1"
        config = json.loads((first_step / "config.json").read_text())
        assert config["privacy_initialization_weights_sha256"] == info["weights_sha256"]
        shutil.rmtree(source)
        set_seed(42)
        resumed = Trainer(model=PrivacyFilterModel(copy.deepcopy(target)), args=args,
                          train_dataset=dataset, data_collator=default_data_collator)
        resumed.train(resume_from_checkpoint=str(first_step))
        assert resumed.state.global_step == 2
        torch.testing.assert_close(resumed.model.score.weight, expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


def test_warmstart_and_same_phase_resume_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        train({}, resume_from_checkpoint="some-checkpoint", initialize_from_checkpoint="other-checkpoint")


def test_full_nvidia_training_refuses_sampled_or_missing_data():
    manifest = {"group_cross_split_count": 0, "sources": [{"id": "nvidia/Nemotron-PII"}],
                "source_parameters": {"nvidia_train_is_complete": True, "sampled_nvidia_holdout": False},
                "postwrite_isolation_audit": {"passed": True},
                "splits": {"train": {"sources": {"nemotron": 1}},
                           "validation": {"sources": {"nemotron": 1}}}}
    validate_training_manifest(manifest, require_complete_nvidia=True)
    sampled = copy.deepcopy(manifest)
    sampled["source_parameters"]["sampled_nvidia_holdout"] = True
    with pytest.raises(ValueError, match="NVIDIA"):
        validate_training_manifest(sampled, require_complete_nvidia=True)
    missing = copy.deepcopy(manifest)
    missing["splits"]["train"]["sources"] = {}
    with pytest.raises(ValueError, match="NVIDIA"):
        validate_training_manifest(missing, require_complete_nvidia=True)


def test_warmstart_requires_holdouts_filtered_against_source_training_dataset():
    initialization = {"source_dataset_manifest_sha256": "prior-dataset"}
    manifest = {"source_parameters": {"baseline_manifest_sha256": "prior-dataset"},
                "postwrite_isolation_audit": {"passed": True}}
    validate_warmstart_lineage(initialization, manifest)
    manifest["source_parameters"]["baseline_manifest_sha256"] = "another-dataset"
    with pytest.raises(ValueError, match="source dataset"):
        validate_warmstart_lineage(initialization, manifest)


def test_real_data_smoke_selects_a_supervised_nvidia_batch(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"source": "openpii", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0, 1]},
            {"source": "nemotron", "input_ids": [3, 4], "attention_mask": [1, 1], "labels": [0, 0]},
            {"source": "nemotron", "input_ids": [5, 6], "attention_mask": [1, 1], "labels": [0, 4]}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    batch = select_nvidia_batch(tmp_path)
    assert batch["input_ids"].tolist() == [[5, 6]]
    assert batch["labels"].tolist() == [[0, 4]]
    with pytest.raises(ValueError, match="NVIDIA"):
        select_nvidia_batch(tmp_path, minimum_tokens=3)
