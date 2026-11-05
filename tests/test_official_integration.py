import os

import pytest

from valendra_privacy_filter.datasets.alignment import align_labels
from valendra_privacy_filter.labels import label_names
from valendra_privacy_filter.training.smoke import MODEL_ID, MODEL_REVISION, official_smoke

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("RUN_OFFICIAL_INTEGRATION") != "1", reason="official checkpoint integration is opt-in")
def test_official_qwen_tokenizer_unicode_offsets():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    text = "💬 Ana García llamó a ana@example.com 🧪"
    spans = []
    for value, label in [("Ana García", "PERSON"), ("ana@example.com", "EMAIL")]:
        start = text.index(value)
        spans.append({"start": start, "end": start + len(value), "label": label})
    tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    names = label_names()
    labels = align_labels(text, tokens["offset_mapping"], spans, [], names)
    for span in spans:
        positions = [offset for offset, tag in zip(tokens["offset_mapping"], labels)
                     if tag != -100 and names[tag].endswith("-" + span["label"])]
        assert positions
        assert text[positions[0][0]:positions[-1][1]].strip() == text[span["start"]:span["end"]]


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_INTEGRATION") != "1", reason="GPU smoke is opt-in")
def test_official_checkpoint_bf16_train_and_safetensors(tmp_path):
    report = official_smoke(tmp_path, device="cuda")
    assert report["checkpoint_reloaded"]
    assert report["backbone_updated"] and report["head_updated"]
    assert report["logits_dtype"] == "torch.bfloat16"
    assert report["optimizer_state_dtype"] == "torch.float32"
