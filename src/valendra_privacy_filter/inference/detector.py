"""Windowed classification and redaction in original Unicode coordinates."""

import math

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
        return {"entities": decode_spans(text, offsets, tags, self.names, probabilities)}

    def redact(self, text):
        return redact_text(text, self.detect(text)["entities"])
