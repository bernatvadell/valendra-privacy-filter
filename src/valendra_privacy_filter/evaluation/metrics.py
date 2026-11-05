"""Document-level exact-span metrics without retaining entity values."""

from collections import Counter, defaultdict


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "f2": f2}


def evaluate_documents(documents):
    total = Counter()
    categories, languages = defaultdict(Counter), defaultdict(Counter)
    confusion = defaultdict(Counter)
    count = negative_count = negative_fp = 0
    for document in documents:
        count += 1
        ignored = document.get("ignored", [])
        gold = {(x["start"], x["end"], x["label"]) for x in document["gold"]}
        pred = {(x["start"], x["end"], x["type"]) for x in document["predicted"]
                if not any(x["start"] < y["end"] and y["start"] < x["end"] for y in ignored)}
        tp, fp, fn = gold & pred, pred - gold, gold - pred
        language = document.get("language", "unknown")
        for key, values in (("tp", tp), ("fp", fp), ("fn", fn)):
            total[key] += len(values)
            languages[language][key] += len(values)
            for _, _, category in values:
                categories[category][key] += 1
        if not gold and not ignored:
            negative_count += 1
            negative_fp += bool(pred)
        for _, _, category in tp:
            confusion[category][category] += 1
        remaining_fp = set(fp)
        for start, end, category in sorted(fn):
            same_boundary = sorted(x for x in remaining_fp if x[:2] == (start, end))
            if same_boundary:
                other = same_boundary[0]
                confusion[category][other[2]] += 1
                remaining_fp.remove(other)
            else:
                confusion[category]["NONE"] += 1
        for _, _, category in remaining_fp:
            confusion["NONE"][category] += 1
    return {
        "documents": count,
        "overall": _metrics(total["tp"], total["fp"], total["fn"]),
        "by_category": {k: _metrics(v["tp"], v["fp"], v["fn"]) for k, v in sorted(categories.items())},
        "by_language": {k: _metrics(v["tp"], v["fp"], v["fn"]) for k, v in sorted(languages.items())},
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
        "false_positives_per_document": total["fp"] / count if count else 0.0,
        "negative_documents": {"count": negative_count, "with_false_positive": negative_fp,
                               "false_positive_rate": negative_fp / negative_count if negative_count else 0.0},
    }
