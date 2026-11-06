"""Windowed classification and redaction in original Unicode coordinates."""

import math
import unicodedata

import torch
from transformers import AutoTokenizer

from valendra_privacy_filter.models import PrivacyFilterModel
from valendra_privacy_filter.models.decode import viterbi_decode


def decode_spans(text, offsets, tags, names, probabilities):
    if not (len(offsets) == len(tags) == len(probabilities)):
        raise ValueError("Token offsets, tags and probabilities must have equal length")
    entities, active = [], None

    def emit(indices, category):
        start = min(offsets[i][0] for i in indices)
        end = max(offsets[i][1] for i in indices)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            score = math.exp(sum(math.log(max(float(probabilities[i]), 1e-30)) for i in indices) / len(indices))
            entities.append({"type": category, "start": start, "end": end,
                             "value": text[start:end], "score": score})

    for i, tag in enumerate(tags):
        name = names[tag]
        if name == "O":
            active = None
            continue
        prefix, category = name.split("-", 1)
        if prefix == "S":
            emit([i], category)
            active = None
        elif prefix == "B":
            active = (category, [i])
        elif prefix in {"I", "E"} and active is not None and active[0] == category:
            active[1].append(i)
            if prefix == "E":
                emit(active[1], category)
                active = None
        else:
            active = None
    # Byte-level offsets can overlap for Unicode code points. Prefer the
    # higher-confidence interval, then earlier and longer ones on ties.
    selected = []
    for entity in sorted(entities, key=lambda e: (-e["score"], e["start"], -(e["end"] - e["start"]))):
        if not any(entity["start"] < prior["end"] and prior["start"] < entity["end"] for prior in selected):
            selected.append(entity)
    return sorted(selected, key=lambda e: (e["start"], e["end"]))


def _word_character(char):
    return char.isalnum() or char == "_" or unicodedata.category(char).startswith("M")


def _reconcile_adjacent_secret_fragments(text, offsets, logits, names, entities):
    """Re-score a fully detected word split into adjacent entity fragments."""
    if len(entities) < 2 or not any(entity["type"] == "SECRET" for entity in entities):
        return entities
    if logits.ndim != 2 or logits.shape != (len(offsets), len(names)):
        return entities
    for offset in offsets:
        if (not isinstance(offset, (tuple, list)) or len(offset) != 2
                or any(type(value) is not int for value in offset)
                or not 0 <= offset[0] <= offset[1] <= len(text)):
            return entities
    label_ids = {name: index for index, name in enumerate(names)}
    result = []
    position = 0
    while position < len(entities):
        end_position = position + 1
        while end_position < len(entities):
            left, right = entities[end_position - 1], entities[end_position]
            boundary = left["end"]
            if (boundary != right["start"] or not 0 < boundary < len(text)
                    or not _word_character(text[boundary - 1]) or not _word_character(text[boundary])):
                break
            end_position += 1
        group = entities[position:end_position]
        start, end = group[0]["start"], group[-1]["end"]
        categories = {entity["type"] for entity in group}
        whole_word = (all(_word_character(char) for char in text[start:end])
                      and (start == 0 or not _word_character(text[start - 1]))
                      and (end == len(text) or not _word_character(text[end])))
        if len(group) > 1 and "SECRET" in categories and whole_word:
            indices = [index for index, (a, b) in enumerate(offsets) if a < end and b > start]
        else:
            indices = []
        if indices:
            first_start = offsets[indices[0]][0]
            last_end = offsets[indices[-1]][1]
            aligned = (indices == list(range(indices[0], indices[-1] + 1))
                       and all(type(offsets[i][0]) is int and type(offsets[i][1]) is int
                               and 0 <= offsets[i][0] < offsets[i][1] <= len(text) for i in indices)
                       and first_start <= start and last_end >= end
                       and not any(not char.isspace() for char in text[first_start:start])
                       and not any(not char.isspace() for char in text[end:last_end]))
            prefixes = (["S"] if len(indices) == 1 else ["B"] + ["I"] * (len(indices) - 2) + ["E"])
            if aligned and all(f"{prefix}-{category}" in label_ids for category in categories for prefix in prefixes):
                log_probabilities = torch.log_softmax(logits[indices], dim=-1)
                scores = {category: sum(float(log_probabilities[i, label_ids[f"{prefix}-{category}"]])
                                        for i, prefix in enumerate(prefixes))
                          for category in categories}
                if (all(math.isfinite(score) for score in scores.values())
                        and all(scores["SECRET"] > score for category, score in scores.items() if category != "SECRET")):
                    result.append({"type": "SECRET", "start": start, "end": end, "value": text[start:end],
                                   "score": math.exp(scores["SECRET"] / len(indices))})
                    position = end_position
                    continue
        result.extend(group)
        position = end_position
    return result


_RUN_OPENING = "\"'«“‘([{<"
_RUN_CLOSING = "\"'»”’)]}>.,;:"


def _whitespace_run(text, start, end):
    """The whitespace-delimited run around a span, without wrapping quotes/brackets or trailing punctuation."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    while start < end and text[start] in _RUN_OPENING:
        start += 1
    while end > start and text[end - 1] in _RUN_CLOSING:
        end -= 1
    return start, end


def _extend_secrets_to_whitespace_runs(text, offsets, logits, names, entities):
    """Grow a SECRET fragment to its whitespace-delimited run when the model prefers that reading.

    Credentials often contain symbols (``Barca1983#``, ``ghp_…``, ``Avalon_2026$``) and the model may tag only part
    of them, leaving the rest as ``O``. The whole run is scored as a single SECRET and replaces the fragments only when
    that path scores strictly higher than reading every token as ``O`` and than each other category found in the run.
    """
    if not any(entity["type"] == "SECRET" for entity in entities):
        return entities
    if logits.ndim != 2 or logits.shape != (len(offsets), len(names)):
        return entities
    for offset in offsets:
        if (not isinstance(offset, (tuple, list)) or len(offset) != 2
                or any(type(value) is not int for value in offset)
                or not 0 <= offset[0] <= offset[1] <= len(text)):
            return entities
    label_ids = {name: index for index, name in enumerate(names)}
    if "O" not in label_ids:
        return entities
    log_probabilities = torch.log_softmax(logits, dim=-1)
    runs = {}
    for entity in entities:
        if entity["type"] != "SECRET":
            continue
        start, end = _whitespace_run(text, entity["start"], entity["end"])
        if (start, end) in runs or (start, end) == (entity["start"], entity["end"]):
            continue
        if start > entity["start"] or end < entity["end"]:
            continue
        overlapping = [item for item in entities if item["start"] < end and item["end"] > start]
        if any(item["start"] < start or item["end"] > end for item in overlapping):
            continue
        indices = [index for index, (a, b) in enumerate(offsets) if a < end and b > start]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            continue
        if (any(not char.isspace() for char in text[offsets[indices[0]][0]:start])
                or any(not char.isspace() for char in text[end:offsets[indices[-1]][1]])):
            continue
        prefixes = ["S"] if len(indices) == 1 else ["B"] + ["I"] * (len(indices) - 2) + ["E"]
        categories = {item["type"] for item in overlapping}
        if not all(f"{prefix}-{category}" in label_ids for category in categories for prefix in prefixes):
            continue
        scores = {category: sum(float(log_probabilities[i, label_ids[f"{prefix}-{category}"]])
                                for i, prefix in zip(indices, prefixes))
                  for category in categories}
        outside = sum(float(log_probabilities[i, label_ids["O"]]) for i in indices)
        secret = scores["SECRET"]
        if (math.isfinite(secret) and secret > outside
                and all(secret > score for category, score in scores.items() if category != "SECRET")):
            runs[(start, end)] = math.exp(secret / len(indices))
    if not runs:
        return entities
    kept = [item for item in entities if not any(item["start"] < end and item["end"] > start for start, end in runs)]
    grown = [{"type": "SECRET", "start": start, "end": end, "value": text[start:end], "score": score}
             for (start, end), score in runs.items()]
    return sorted(kept + grown, key=lambda item: (item["start"], item["end"]))


def redact_text(text, entities):
    result = text
    last_start = len(text)
    for entity in sorted(entities, key=lambda item: (item["start"], item["end"]), reverse=True):
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= last_start or text[start:end] != entity["value"]:
            raise ValueError("Invalid or overlapping entity offsets")
        result = result[:start] + f"[{entity['type']}]" + result[end:]
        last_start = start
    return result


class Detector:
    def __init__(self, model, tokenizer, max_length=512, overlap=64, device="cpu", precision="fp32", temperature=1.0):
        if max_length < 2 or not 0 <= overlap < max_length:
            raise ValueError("Require max_length >= 2 and 0 <= overlap < max_length")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.device = torch.device(device)
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if precision == "bf16" and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support BF16")
        self.model = model.to(device=self.device, dtype=torch.float32 if precision == "fp32" else torch.bfloat16).eval()
        self.tokenizer = tokenizer
        self.max_length, self.overlap, self.temperature = max_length, overlap, temperature
        mapping = self.model.config.id2label
        self.names = [mapping[i] if i in mapping else mapping[str(i)] for i in range(len(mapping))]

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = PrivacyFilterModel.from_pretrained(path, use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with character offsets is required")
        return cls(model, tokenizer, **kwargs)

    @torch.inference_mode()
    def detect(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return {"entities": []}
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if not ids:
            return {"entities": []}
        totals = torch.zeros((len(ids), len(self.names)), dtype=torch.float32)
        counts = torch.zeros((len(ids), 1), dtype=torch.float32)
        step = self.max_length - self.overlap
        for start in range(0, len(ids), step):
            stop = min(start + self.max_length, len(ids))
            tokens = torch.tensor([ids[start:stop]], device=self.device, dtype=torch.long)
            output = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
            totals[start:stop] += output.logits[0].float().cpu()
            counts[start:stop] += 1
            if stop == len(ids):
                break
        logits = totals / counts / self.temperature
        tags = viterbi_decode(logits, self.names)
        probabilities = torch.softmax(logits, dim=-1).gather(1, torch.tensor(tags)[:, None]).squeeze(1).tolist()
        entities = decode_spans(text, offsets, tags, self.names, probabilities)
        entities = _reconcile_adjacent_secret_fragments(text, offsets, logits, self.names, entities)
        return {"entities": _extend_secrets_to_whitespace_runs(text, offsets, logits, self.names, entities)}

    def redact(self, text):
        return redact_text(text, self.detect(text)["entities"])
