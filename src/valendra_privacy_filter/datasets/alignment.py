"""BIOES labels from Unicode character intervals and fast-tokenizer offsets."""

from collections import Counter


def align_labels(text, offsets, spans, ignored, names, return_diagnostics=False):
    lookup = {name: index for index, name in enumerate(names)}
    labels = [-100 if start == end else lookup["O"] for start, end in offsets]

    def indices(span):
        return [i for i, (start, end) in enumerate(offsets)
                if start < end and start < span["end"] and end > span["start"]]

    # A flat BIOES sequence cannot faithfully encode intersecting annotations.
    conflicts = set()
    for i, left in enumerate(spans):
        for j in range(i + 1, len(spans)):
            right = spans[j]
            if left["start"] < right["end"] and right["start"] < left["end"]:
                conflicts.update((i, j))
    ignored_tokens = {i for item in ignored for i in indices(item)}
    diagnostics = Counter()
    for n, item in enumerate(spans):
        positions = indices(item)
        if not positions:
            continue
        first, last = offsets[positions[0]], offsets[positions[-1]]
        # Partial entities or non-whitespace overhang cannot be reconstructed.
        representable = (first[0] <= item["start"] and last[1] >= item["end"]
                         and not text[first[0]:item["start"]].strip()
                         and not text[item["end"]:last[1]].strip()
                         and text[item["start"]:item["end"]] == text[item["start"]:item["end"]].strip())
        if n in conflicts:
            diagnostics["overlapping_entities"] += 1
        elif not representable:
            diagnostics["unrepresentable_or_window_boundary"] += 1
        elif ignored_tokens.intersection(positions):
            diagnostics["overlaps_ignored_annotation"] += 1
        if n in conflicts or not representable or ignored_tokens.intersection(positions):
            ignored_tokens.update(positions)
            continue
        if len(positions) == 1:
            labels[positions[0]] = lookup[f"S-{item['label']}"]
        else:
            for j, position in enumerate(positions):
                prefix = "B" if j == 0 else "E" if j == len(positions) - 1 else "I"
                labels[position] = lookup[f"{prefix}-{item['label']}"]
    for position in ignored_tokens:
        labels[position] = -100
    diagnostics["masked_annotation_tokens"] = len(ignored_tokens)
    return (labels, dict(diagnostics)) if return_diagnostics else labels
