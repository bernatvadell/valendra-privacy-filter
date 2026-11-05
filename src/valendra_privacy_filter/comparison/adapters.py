"""Inference-only adapters with strict original-text offsets and no implicit truncation."""

import math
import warnings


class ComparisonAdapterError(ValueError):
    """Sanitized comparison failure that never contains input text."""


def _prediction(text, start, end, value, label, score):
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
            or not isinstance(value, str) or text[start:end] != value or not isinstance(label, str) or not label):
        raise ComparisonAdapterError("invalid_prediction_offsets_or_label")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            raise ComparisonAdapterError("invalid_prediction_score") from None
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ComparisonAdapterError("invalid_prediction_score")
    return {"start": start, "end": end, "value": value, "label": label, "score": score}


class ValendraAdapter:
    def __init__(self, detector):
        self.detector = detector

    def detect(self, text):
        return [_prediction(text, e.get("start"), e.get("end"), e.get("value"), e.get("type"), e.get("score"))
                for e in self.detector.detect(text)["entities"]]

    def count_tokens(self, text):
        return len(self.detector.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


class OpenAIAdapter:
    def __init__(self, redactor):
        self.redactor = redactor

    def detect(self, text):
        result = self.redactor.redact(text)
        if isinstance(result, str):
            raise ComparisonAdapterError("openai_typed_output_required")
        if getattr(result, "text", None) != text or getattr(result, "summary", {}).get("decoded_mismatch", False):
            raise ComparisonAdapterError("openai_tokenizer_roundtrip_mismatch")
        return [_prediction(text, span.start, span.end, span.text, span.label, None)
                for span in result.detected_spans]

    def count_tokens(self, text):
        return len(self.redactor.get_runtime().encoding.encode(text, allowed_special="all"))


class GLiNERAdapter:
    def __init__(self, model, labels, threshold=0.3, overlap_words=32):
        if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("GLiNER labels must be distinct, nonempty strings")
        if not 0 <= threshold <= 1 or type(overlap_words) is not int or overlap_words < 0:
            raise ValueError("Invalid GLiNER threshold or overlap")
        self.model, self.labels = model, list(labels)
        self.threshold, self.overlap_words = float(threshold), overlap_words
        self.last_tokens = self.last_windows = 0
        if len(self.labels) > int(model.config.max_types):
            raise ValueError("GLiNER label count exceeds model configuration")
        self.max_words = int(model.config.max_len)
        encoder = model.config.encoder_config
        limit = encoder.get("max_position_embeddings") if isinstance(encoder, dict) else getattr(encoder, "max_position_embeddings", None)
        tokenizer_limit = getattr(model.data_processor.transformer_tokenizer, "model_max_length", None)
        if type(limit) is not int or limit < 1 or self.max_words < 1:
            raise ValueError("GLiNER encoder limits are unavailable")
        if type(tokenizer_limit) is int and 0 < tokenizer_limit < 1_000_000:
            limit = min(limit, tokenizer_limit)
        self.max_tokens = limit

    def _token_count(self, words):
        processor = self.model.data_processor
        prepared, _ = processor.prepare_inputs([words], self.labels)
        encoded = processor.transformer_tokenizer(prepared, is_split_into_words=True, truncation=False,
                                                  padding=False, add_special_tokens=True)
        token_ids = encoded["input_ids"]
        return len(token_ids[0])

    def count_tokens(self, text):
        return len(self.model.data_processor.transformer_tokenizer(text, add_special_tokens=False,
                                                                    truncation=False)["input_ids"])

    def detect(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self.last_tokens = self.last_windows = 0
        token_batches, starts_batches, ends_batches = self.model.prepare_inputs([text])
        words, starts, ends = token_batches[0], starts_batches[0], ends_batches[0]
        if not (len(words) == len(starts) == len(ends)):
            raise ComparisonAdapterError("gliner_invalid_token_offsets")
        for index, (word, start, end) in enumerate(zip(words, starts, ends)):
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or text[start:end] != word or (index and start < ends[index - 1])):
                raise ComparisonAdapterError("gliner_invalid_token_offsets")
        if not words:
            return []
        candidates = {}
        start_index = 0
        while start_index < len(words):
            low, high, best = start_index + 1, min(len(words), start_index + self.max_words), start_index
            while low <= high:
                middle = (low + high) // 2
                if self._token_count(words[start_index:middle]) <= self.max_tokens:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best == start_index:
                raise ComparisonAdapterError("gliner_window_capacity_insufficient")
            start = 0 if start_index == 0 else starts[start_index]
            end = len(text) if best == len(words) else ends[best - 1]
            fragment = text[start:end]
            fragment_words = self.model.prepare_inputs([fragment])[0][0]
            if fragment_words != words[start_index:best]:
                raise ComparisonAdapterError("gliner_window_tokenization_mismatch")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                predicted = self.model.predict_entities(fragment, self.labels, threshold=self.threshold,
                                                         flat_ner=True, multi_label=False)
            if any("truncat" in str(w.message).lower() for w in caught):
                raise ComparisonAdapterError("gliner_unexpected_truncation")
            self.last_tokens += self._token_count(words[start_index:best])
            self.last_windows += 1
            for item in predicted:
                raw_start, raw_end = item.get("start"), item.get("end")
                if (type(raw_start) is not int or type(raw_end) is not int
                        or not 0 <= raw_start < raw_end <= len(fragment)):
                    raise ComparisonAdapterError("invalid_prediction_offsets_or_label")
                if item.get("score") is None:
                    raise ComparisonAdapterError("invalid_prediction_score")
                candidate = _prediction(text, start + raw_start, start + raw_end, item.get("text"),
                                        item.get("label"), item.get("score"))
                if candidate["label"] not in self.labels:
                    raise ComparisonAdapterError("gliner_unrequested_label")
                key = (candidate["start"], candidate["end"], candidate["label"])
                if key not in candidates or candidate["score"] > candidates[key]["score"]:
                    candidates[key] = candidate
            if best == len(words):
                break
            if best - start_index <= self.overlap_words:
                raise ComparisonAdapterError("gliner_window_capacity_insufficient_for_overlap")
            start_index = best - self.overlap_words
        selected = []
        for candidate in sorted(candidates.values(), key=lambda e: (-e["score"], e["start"], -(e["end"] - e["start"]), e["label"])):
            if not any(candidate["start"] < other["end"] and other["start"] < candidate["end"] for other in selected):
                selected.append(candidate)
        return sorted(selected, key=lambda e: (e["start"], e["end"], e["label"]))
