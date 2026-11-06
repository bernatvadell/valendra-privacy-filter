from types import SimpleNamespace

import pytest
import torch

from valendra_privacy_filter.inference.detector import (Detector, _extend_secrets_to_whitespace_runs,
                                                        _reconcile_adjacent_secret_fragments,
                                                        decode_spans, redact_text)
from valendra_privacy_filter.labels import label_names
from valendra_privacy_filter.models.decode import viterbi_decode


def test_viterbi_enforces_start_end_and_type_transitions():
    names = label_names(["PERSON", "EMAIL"])
    logits = torch.full((3, len(names)), -10.0)
    logits[0, names.index("I-PERSON")] = 20
    logits[0, names.index("B-PERSON")] = 19
    logits[1, names.index("I-EMAIL")] = 20
    logits[1, names.index("I-PERSON")] = 18
    logits[2, names.index("E-PERSON")] = 20
    path = [names[i] for i in viterbi_decode(logits, names)]
    assert path == ["B-PERSON", "I-PERSON", "E-PERSON"]


def test_unicode_and_adjacent_spans_reconstruct_original_text():
    text = "💬Ana García|x@example.com!"
    names = label_names()
    a, b = text.index("Ana"), text.index("|")
    c, d = b + 1, text.index("!")
    offsets = [(0, 1), (a, 4), (4, b), (b, c), (c, d), (d, d + 1)]
    tags = ["O", "B-PERSON", "E-PERSON", "O", "S-EMAIL", "O"]
    entities = decode_spans(text, offsets, [names.index(x) for x in tags], names, [1.0] * len(tags))
    assert [(e["type"], e["value"], e["start"], e["end"]) for e in entities] == [
        ("PERSON", "Ana García", a, b), ("EMAIL", "x@example.com", c, d)
    ]
    assert redact_text(text, entities) == "💬[PERSON]|[EMAIL]!"


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class CharacterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.names = label_names()
        self.config = SimpleNamespace(id2label=dict(enumerate(self.names)))

    def forward(self, input_ids, attention_mask):
        scores = torch.full((*input_ids.shape, len(self.names)), -12.0, device=input_ids.device)
        scores[:, :, 0] = 12.0
        for char, tag in [("A", "B-PERSON"), ("d", "I-PERSON"), ("a", "E-PERSON")]:
            selected = input_ids == ord(char)
            scores[:, :, 0][selected] = -12.0
            scores[:, :, self.names.index(tag)][selected] = 12.0
        return SimpleNamespace(logits=scores)


def test_detector_merges_overlapping_windows_and_redacts():
    detector = Detector(CharacterModel(), CharacterTokenizer(), max_length=4, overlap=2)
    text = "💬Hi Ada!"
    result = detector.detect(text)
    assert [(x["type"], x["start"], x["end"], x["value"]) for x in result["entities"]] == [
        ("PERSON", 4, 7, "Ada")
    ]
    assert detector.redact(text) == "💬Hi [PERSON]!"


class FragmentedWordModel(torch.nn.Module):
    def __init__(self, *, prefer_person=False, second_type="SECRET"):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.names = label_names()
        self.config = SimpleNamespace(id2label=dict(enumerate(self.names)))
        self.prefer_person = prefer_person
        self.second_type = second_type

    def forward(self, input_ids, attention_mask):
        scores = torch.full((*input_ids.shape, len(self.names)), -12.0, device=input_ids.device)
        scores[:, :, 0] = 8.0
        assignments = {
            "A": {"B-PERSON": 8, "B-SECRET": -2 if self.prefer_person else 7},
            "Á": {"B-PERSON": 8, "B-SECRET": -2 if self.prefer_person else 7},
            "b": {"E-PERSON": 8, "I-SECRET": -2 if self.prefer_person else 6, "I-PERSON": 7 if self.prefer_person else -12},
            "1": {f"B-{self.second_type}": 8, "I-SECRET": -2 if self.prefer_person else 7, "I-PERSON": 7 if self.prefer_person else -12},
            "2": {f"E-{self.second_type}": 8, "E-PERSON": 7 if self.prefer_person else -12},
        }
        for char, labels in assignments.items():
            selected = input_ids == ord(char)
            scores[:, :, 0][selected] = -12.0
            for label, value in labels.items():
                scores[:, :, self.names.index(label)][selected] = value
        return SimpleNamespace(logits=scores)


def test_detector_reconciles_mixed_fragments_using_the_joint_secret_score():
    model = FragmentedWordModel()
    detector = Detector(model, CharacterTokenizer())
    text = "Ab12"
    token_ids = torch.tensor([[ord(char) for char in text]])
    logits = model(token_ids, torch.ones_like(token_ids)).logits[0]
    assert [model.names[i] for i in viterbi_decode(logits, model.names)] == [
        "B-PERSON", "E-PERSON", "B-SECRET", "E-SECRET"
    ]
    secret_path = [model.names.index(label) for label in ("B-SECRET", "I-SECRET", "I-SECRET", "E-SECRET")]
    log_probabilities = torch.log_softmax(logits, dim=-1)
    expected_score = torch.exp(torch.stack([log_probabilities[i, tag] for i, tag in enumerate(secret_path)]).mean()).item()

    result = detector.detect(text)["entities"]
    assert [(item["type"], item["start"], item["end"], item["value"]) for item in result] == [
        ("SECRET", 0, 4, text)
    ]
    assert result[0]["score"] == pytest.approx(expected_score)
    assert detector.redact(text) == "[SECRET]"


def test_detector_reconciles_an_alphanumeric_unicode_word_without_moving_offsets():
    detector = Detector(FragmentedWordModel(), CharacterTokenizer())
    text = "💬Áb12!"
    entities = detector.detect(text)["entities"]
    assert [(item["type"], item["start"], item["end"], item["value"]) for item in entities] == [
        ("SECRET", 1, 5, "Áb12")
    ]
    assert detector.redact(text) == "💬[SECRET]!"


def test_detector_does_not_merge_separate_words():
    detector = Detector(FragmentedWordModel(), CharacterTokenizer())
    assert [(item["type"], item["value"]) for item in detector.detect("Ab 12")["entities"]] == [
        ("PERSON", "Ab"), ("SECRET", "12")
    ]


@pytest.mark.parametrize("text", ["Ab-12", "xAb12", "Ab12x"])
def test_detector_extends_fragments_to_the_whitespace_run_when_secret_scores_higher(text):
    detector = Detector(FragmentedWordModel(), CharacterTokenizer())
    assert [(item["type"], item["value"]) for item in detector.detect(text)["entities"]] == [("SECRET", text)]


@pytest.mark.parametrize("text", ["Ab-12", "xAb12", "Ab12x"])
def test_detector_keeps_run_fragments_when_another_category_scores_higher(text):
    detector = Detector(FragmentedWordModel(prefer_person=True), CharacterTokenizer())
    assert [(item["type"], item["value"]) for item in detector.detect(text)["entities"]] == [
        ("PERSON", "Ab"), ("SECRET", "12")
    ]


class SecretTailModel(torch.nn.Module):
    """Digits are clearly SECRET; other characters lean towards O by ``prefix_margin``."""

    def __init__(self, prefix_margin):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.names = label_names()
        self.config = SimpleNamespace(id2label=dict(enumerate(self.names)))
        self.prefix_margin = prefix_margin

    def forward(self, input_ids, attention_mask):
        scores = torch.full((*input_ids.shape, len(self.names)), -12.0, device=input_ids.device)
        scores[:, :, 0] = 6.0
        for label in ("B-SECRET", "I-SECRET", "E-SECRET"):
            scores[:, :, self.names.index(label)] = 6.0 - self.prefix_margin
        digits = (input_ids >= ord("0")) & (input_ids <= ord("9"))
        scores[:, :, 0][digits] = -12.0
        for label in ("B-SECRET", "I-SECRET", "E-SECRET"):
            scores[:, :, self.names.index(label)][digits] = 8.0
        return SimpleNamespace(logits=scores)


@pytest.mark.parametrize("text, value, redacted", [
    ("token ghp_12 ok", "ghp_12", "token [SECRET] ok"),
    ("clave Avalon_2026$", "Avalon_2026$", "clave [SECRET]"),
    ('usa "key#12", gracias', "key#12", 'usa "[SECRET]", gracias'),
])
def test_detector_absorbs_weak_outside_tokens_of_a_secret_run(text, value, redacted):
    detector = Detector(SecretTailModel(prefix_margin=1.0), CharacterTokenizer())
    entities = detector.detect(text)["entities"]
    assert [(item["type"], item["value"], text[item["start"]:item["end"]]) for item in entities] == [
        ("SECRET", value, value)
    ]
    assert detector.redact(text) == redacted


def test_detector_does_not_absorb_tokens_the_model_clearly_reads_as_outside():
    detector = Detector(SecretTailModel(prefix_margin=10.0), CharacterTokenizer())
    assert [(item["type"], item["value"]) for item in detector.detect("id execution-12 ok")["entities"]] == [
        ("SECRET", "12")
    ]


def test_secret_runs_reject_out_of_bounds_token_offsets():
    text = "ab12"
    model = SecretTailModel(prefix_margin=1.0)
    tokens = torch.tensor([[ord(char) for char in text]])
    logits = model(tokens, torch.ones_like(tokens)).logits[0]
    entities = [{"type": "SECRET", "start": 2, "end": 4, "value": "12", "score": 0.9}]
    offsets = [(0, 1), (1, 2), (2, 3), (3, 5)]
    assert _extend_secrets_to_whitespace_runs(text, offsets, logits, model.names, entities) == entities


def test_detector_keeps_fragments_when_another_joint_category_scores_higher():
    detector = Detector(FragmentedWordModel(prefer_person=True), CharacterTokenizer())
    assert [(item["type"], item["value"]) for item in detector.detect("Ab12")["entities"]] == [
        ("PERSON", "Ab"), ("SECRET", "12")
    ]


def test_detector_does_not_merge_adjacent_fragments_without_a_secret():
    detector = Detector(FragmentedWordModel(second_type="EMAIL"), CharacterTokenizer())
    assert [(item["type"], item["value"]) for item in detector.detect("Ab12")["entities"]] == [
        ("PERSON", "Ab"), ("EMAIL", "12")
    ]


def test_secret_reconciliation_rejects_out_of_bounds_token_offsets():
    text = "Ab12"
    model = FragmentedWordModel()
    tokens = torch.tensor([[ord(char) for char in text]])
    logits = model(tokens, torch.ones_like(tokens)).logits[0]
    entities = [{"type": "PERSON", "start": 0, "end": 2, "value": "Ab", "score": 0.9},
                {"type": "SECRET", "start": 2, "end": 4, "value": "12", "score": 0.9}]
    offsets = [(0, 1), (1, 2), (2, 3), (3, 5)]
    assert _reconcile_adjacent_secret_fragments(text, offsets, logits, model.names, entities) == entities


def test_secret_reconciliation_keeps_fragments_when_candidate_scores_tie():
    text = "Ab12"
    names = label_names()
    logits = torch.zeros((4, len(names)))
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]
    entities = [{"type": "PERSON", "start": 0, "end": 2, "value": "Ab", "score": 0.9},
                {"type": "SECRET", "start": 2, "end": 4, "value": "12", "score": 0.9}]
    assert _reconcile_adjacent_secret_fragments(text, offsets, logits, names, entities) == entities


def test_secret_reconciliation_accepts_leading_space_in_a_token_offset():
    text = " Ab12!"
    model = FragmentedWordModel()
    tokens = torch.tensor([[ord(char) for char in "Ab12"]])
    logits = model(tokens, torch.ones_like(tokens)).logits[0]
    offsets = [(0, 2), (2, 3), (3, 4), (4, 5)]
    entities = [{"type": "PERSON", "start": 1, "end": 3, "value": "Ab", "score": 0.9},
                {"type": "SECRET", "start": 3, "end": 5, "value": "12", "score": 0.9}]
    result = _reconcile_adjacent_secret_fragments(text, offsets, logits, model.names, entities)
    assert [(item["type"], item["start"], item["end"], item["value"]) for item in result] == [
        ("SECRET", 1, 5, "Ab12")
    ]


def test_secret_reconciliation_rejects_token_offsets_including_punctuation():
    text = "Ab12!"
    model = FragmentedWordModel()
    tokens = torch.tensor([[ord(char) for char in "Ab12"]])
    logits = model(tokens, torch.ones_like(tokens)).logits[0]
    offsets = [(0, 1), (1, 2), (2, 3), (3, 5)]
    entities = [{"type": "PERSON", "start": 0, "end": 2, "value": "Ab", "score": 0.9},
                {"type": "SECRET", "start": 2, "end": 4, "value": "12", "score": 0.9}]
    assert _reconcile_adjacent_secret_fragments(text, offsets, logits, model.names, entities) == entities


def test_secret_reconciliation_skips_noninteger_token_offsets():
    text = "Ab12"
    model = FragmentedWordModel()
    tokens = torch.tensor([[ord(char) for char in text]])
    logits = model(tokens, torch.ones_like(tokens)).logits[0]
    offsets = [(0, 1), (1, 2), ("2", 3), (3, 4)]
    entities = [{"type": "PERSON", "start": 0, "end": 2, "value": "Ab", "score": 0.9},
                {"type": "SECRET", "start": 2, "end": 4, "value": "12", "score": 0.9}]
    assert _reconcile_adjacent_secret_fragments(text, offsets, logits, model.names, entities) == entities
