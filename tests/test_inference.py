from types import SimpleNamespace

import torch

from valendra_privacy_filter.inference.detector import Detector, decode_spans, redact_text
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
