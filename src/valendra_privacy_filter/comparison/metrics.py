"""Strict character-span comparison without retaining input text or entity values."""

from collections import Counter, defaultdict
import re

from valendra_privacy_filter.evaluation.metrics import _metrics
from valendra_privacy_filter.labels import DEFAULT_CATEGORIES

COMMON_CATEGORIES = frozenset({"PERSON", "EMAIL", "PHONE", "ADDRESS", "URL", "SECRET"})


def _safe_group(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) else "unknown"


def _interval(item, text, *, prediction=False):
    start, end = item.get("start"), item.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("invalid prediction offsets" if prediction else "invalid annotation offsets")
    if prediction and (not isinstance(item.get("value"), str) or item["value"] != text[start:end]):
        raise ValueError("invalid prediction value or offsets")
    return start, end


class ComparisonMetrics:
    def __init__(self, label_map, common_categories=COMMON_CATEGORIES):
        if any(not isinstance(key, str) or not key for key in label_map):
            raise ValueError("Invalid native label mapping")
        if any(value is not None and value not in DEFAULT_CATEGORIES for value in label_map.values()):
            raise ValueError("Invalid target category")
        self.label_map = dict(label_map)
        self.common = frozenset(common_categories)
        if not self.common or not self.common.issubset(DEFAULT_CATEGORIES):
            raise ValueError("Invalid common categories")
        self.documents = self.ignored_predictions = self.negative_count = self.negative_fp = 0
        self.untyped, self.typed = Counter(), Counter()
        self.by_language, self.by_source = defaultdict(lambda: {"untyped": Counter(), "typed_common": Counter()}), defaultdict(lambda: {"untyped": Counter(), "typed_common": Counter()})
        self.category_recall, self.unmapped, self.excluded_gold = defaultdict(Counter), Counter(), Counter()
        self.typed_by_category = defaultdict(Counter)

    @staticmethod
    def _add_counts(counter, tp, fp, fn):
        counter.update({"tp": tp, "fp": fp, "fn": fn})

    def add(self, document):
        text = document.get("text")
        if not isinstance(text, str):
            raise ValueError("invalid document text")
        gold = set()
        for item in document.get("gold", []):
            start, end = _interval(item, text)
            label = item.get("label")
            if label not in DEFAULT_CATEGORIES:
                raise ValueError("invalid annotation label")
            gold.add((start, end, label))
        ignored = [_interval(item, text) for item in document.get("ignored", [])]
        predicted = set()
        for item in document.get("predicted", []):
            start, end = _interval(item, text, prediction=True)
            raw_label = item.get("label")
            if not isinstance(raw_label, str) or not raw_label:
                raise ValueError("invalid prediction label")
            if any(start < ignored_end and ignored_start < end for ignored_start, ignored_end in ignored):
                self.ignored_predictions += 1
                continue
            if raw_label not in self.label_map:
                self.unmapped["__unrecognized__"] += 1
                mapped = None
            else:
                mapped = self.label_map[raw_label]
                if mapped is None:
                    self.unmapped[raw_label] += 1
            predicted.add((start, end, mapped))
        gold_bounds = {(start, end) for start, end, _ in gold}
        prediction_bounds = {(start, end) for start, end, _ in predicted}
        gold_typed = {(start, end, label) for start, end, label in gold if label in self.common}
        predicted_typed = {(start, end, label) for start, end, label in predicted if label in self.common}
        for key, spans in (("tp", gold_typed & predicted_typed), ("fp", predicted_typed - gold_typed),
                           ("fn", gold_typed - predicted_typed)):
            for _, _, category in spans:
                self.typed_by_category[category][key] += 1
        for _, _, label in gold:
            if label not in self.common:
                self.excluded_gold[label] += 1
        for start, end, label in gold:
            self.category_recall[label]["tp" if (start, end) in prediction_bounds else "fn"] += 1
        language, source = _safe_group(document.get("language")), _safe_group(document.get("source"))
        for key, actual, detected in (("untyped", gold_bounds, prediction_bounds),
                                      ("typed_common", gold_typed, predicted_typed)):
            tp, fp, fn = len(actual & detected), len(detected - actual), len(actual - detected)
            self._add_counts(self.untyped if key == "untyped" else self.typed, tp, fp, fn)
            self._add_counts(self.by_language[language][key], tp, fp, fn)
            self._add_counts(self.by_source[source][key], tp, fp, fn)
        self.documents += 1
        if not gold and not ignored:
            self.negative_count += 1
            self.negative_fp += bool(prediction_bounds)

    @staticmethod
    def _format(counter):
        return _metrics(counter["tp"], counter["fp"], counter["fn"])

    def report(self):
        breakdown = lambda groups: {name: {key: self._format(value) for key, value in metrics.items()}
                                     for name, metrics in sorted(groups.items())}
        return {"documents": self.documents, "untyped": self._format(self.untyped),
                "typed_common": self._format(self.typed), "common_categories": sorted(self.common),
                "typed_by_category": {name: self._format(self.typed_by_category[name]) for name in sorted(self.common)},
                "untyped_recall_by_gold_category": {
                    name: {"matched": value["tp"], "total": value["tp"] + value["fn"],
                           "recall": value["tp"] / (value["tp"] + value["fn"])}
                    for name, value in sorted(self.category_recall.items())},
                "by_language": breakdown(self.by_language), "by_source": breakdown(self.by_source),
                "typed_excluded_gold": dict(self.excluded_gold), "unmapped_predictions": dict(self.unmapped),
                "ignored_predictions": self.ignored_predictions,
                "false_positives_per_document": self.untyped["fp"] / self.documents if self.documents else 0.0,
                "negative_documents": {"count": self.negative_count, "with_false_positive": self.negative_fp,
                                       "false_positive_rate": self.negative_fp / self.negative_count if self.negative_count else 0.0}}
