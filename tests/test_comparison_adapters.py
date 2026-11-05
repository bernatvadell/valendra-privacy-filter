import re
from types import SimpleNamespace

import pytest

from valendra_privacy_filter.comparison.adapters import (ComparisonAdapterError, GLiNERAdapter,
                                                        OpenAIAdapter, ValendraAdapter)


def test_openai_preserves_native_labels_and_does_not_invent_scores():
    class FakeOPF:
        def redact(self, text):
            return SimpleNamespace(text=text, summary={"decoded_mismatch": False}, detected_spans=[
                SimpleNamespace(start=2, end=5, text="Ana", label="private_person")])

        def get_runtime(self):
            return SimpleNamespace(encoding=SimpleNamespace(encode=lambda text, allowed_special: list(text)))

    assert OpenAIAdapter(FakeOPF()).detect("👋 Ana!") == [
        {"start": 2, "end": 5, "value": "Ana", "label": "private_person", "score": None}]
    assert OpenAIAdapter(FakeOPF()).count_tokens("👋 Ana!") == 6


def test_openai_rejects_roundtrip_mismatch_without_logging_input():
    class FakeOPF:
        def redact(self, text):
            return SimpleNamespace(text="changed", summary={"decoded_mismatch": True}, detected_spans=[])

    with pytest.raises(ComparisonAdapterError, match="roundtrip") as error:
        OpenAIAdapter(FakeOPF()).detect("sensitive example")
    assert "sensitive example" not in str(error.value)


def test_valendra_adapter_converts_output_without_changing_offsets():
    class FakeDetector:
        def detect(self, text):
            return {"entities": [{"start": 0, "end": 3, "value": "Ana", "type": "PERSON", "score": 0.7}]}

    assert ValendraAdapter(FakeDetector()).detect("Ana") == [
        {"start": 0, "end": 3, "value": "Ana", "label": "PERSON", "score": 0.7}]


class FakeTokenizer:
    model_max_length = 10

    def __call__(self, inputs, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[0] * (len(inputs[0]) + 2)]}


class FakeProcessor:
    def __init__(self):
        self.transformer_tokenizer = FakeTokenizer()

    def prepare_inputs(self, texts, labels):
        prompt = [word for label in labels for word in ("<<ENT>>", label)] + ["<<SEP>>"]
        return [prompt + texts[0]], [len(prompt)]


class FakeGLiNER:
    def __init__(self):
        self.config = SimpleNamespace(max_len=4, max_types=25,
                                      encoder_config=SimpleNamespace(max_position_embeddings=10))
        self.data_processor = FakeProcessor()
        self.fragments = []

    def prepare_inputs(self, texts):
        token_batches, starts, ends = [], [], []
        for text in texts:
            matches = list(re.finditer(r"\S+", text))
            token_batches.append([m.group() for m in matches])
            starts.append([m.start() for m in matches])
            ends.append([m.end() for m in matches])
        return token_batches, starts, ends

    def predict_entities(self, text, labels, **kwargs):
        self.fragments.append(text)
        assert len(text.split()) <= self.config.max_len
        assert len(self.data_processor.prepare_inputs([text.split()], labels)[0][0]) + 2 <= 10
        target = "Ana García"
        if target not in text:
            return []
        start = text.index(target)
        return [{"start": start, "end": start + len(target), "text": target, "label": "name", "score": 0.9}]


def test_gliner_overlapping_windows_cover_unicode_entity_and_deduplicate():
    model = FakeGLiNER()
    detector = GLiNERAdapter(model, ["name"], threshold=0.3, overlap_words=2)
    text = "🌟 uno dos Ana García cuatro cinco seis siete"
    result = detector.detect(text)
    start = text.index("Ana García")
    assert result == [{"start": start, "end": start + 10, "value": "Ana García", "label": "name", "score": 0.9}]
    assert len(model.fragments) > 1


def test_gliner_fails_if_a_window_cannot_respect_encoder_limit():
    model = FakeGLiNER()
    model.config.encoder_config.max_position_embeddings = 5
    detector = GLiNERAdapter(model, ["name"], threshold=0.3, overlap_words=2)
    with pytest.raises(ComparisonAdapterError, match="capacity"):
        detector.detect("uno dos tres cuatro cinco")


def test_gliner_rejects_predictions_outside_their_own_window():
    class Invalid(FakeGLiNER):
        def predict_entities(self, text, labels, **kwargs):
            return [{"start": -1, "end": 1, "text": "xx", "label": "name", "score": 0.9}]

    with pytest.raises(ComparisonAdapterError, match="offsets"):
        GLiNERAdapter(Invalid(), ["name"], overlap_words=2).detect("hello world")


def test_gliner_rejects_missing_scores_and_truncation_warnings():
    import warnings

    class MissingScore(FakeGLiNER):
        def predict_entities(self, text, labels, **kwargs):
            return [{"start": 0, "end": 3, "text": text[:3], "label": "name"}]

    with pytest.raises(ComparisonAdapterError, match="score"):
        GLiNERAdapter(MissingScore(), ["name"], overlap_words=2).detect("Ana")

    class Truncated(FakeGLiNER):
        def predict_entities(self, text, labels, **kwargs):
            warnings.warn("truncated " + text)
            return []

    with pytest.raises(ComparisonAdapterError, match="truncation") as error:
        GLiNERAdapter(Truncated(), ["name"], overlap_words=2).detect("Ana")
    assert "Ana" not in str(error.value)
