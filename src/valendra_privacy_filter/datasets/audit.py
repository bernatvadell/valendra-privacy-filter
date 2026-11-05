"""Aggregate-only checks of stored Unicode spans, BIOES labels and token offsets."""

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from valendra_privacy_filter.datasets.prepare import SPLITS, verify_dataset


def audit_prepared_annotations(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest = verify_dataset(dataset_dir)
    names = manifest["label_names"]
    issues, counts = Counter(), Counter()

    def emit(text, offsets, positions, category, gold):
        start = min(offsets[position][0] for position in positions)
        end = max(offsets[position][1] for position in positions)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        counts["entity_spans_checked"] += 1
        if (start, end, category) not in gold:
            issues["reconstructed_span_mismatch"] += 1

    for split in SPLITS:
        for batch in pq.ParquetFile(dataset_dir / f"{split}.parquet").iter_batches(batch_size=512):
            for row in batch.to_pylist():
                counts["windows_checked"] += 1
                text, labels, offsets = row["text"], row["labels"], row["offset_mapping"]
                if row["window_index"] == 0:
                    counts["documents_checked"] += 1
                    for annotation in row["raw_annotations"]:
                        start, end = annotation["start"], annotation["end"]
                        if not 0 <= start < end <= len(text) or text[start:end] != annotation["value"]:
                            issues["raw_span_value_mismatch"] += 1
                if not (len(labels) == len(offsets) == len(row["input_ids"]) == len(row["attention_mask"])):
                    issues["token_array_length_mismatch"] += 1
                    continue
                gold = {(item["start"], item["end"], item["label"]) for item in row["spans"]}
                active = None
                for position, (label, offset, attended) in enumerate(zip(labels, offsets, row["attention_mask"])):
                    if len(offset) != 2 or not 0 <= offset[0] <= offset[1] <= len(text):
                        issues["invalid_token_offset"] += 1
                        active = None
                        continue
                    if label == -100:
                        if active is not None:
                            issues["invalid_bioes_transition"] += 1
                        active = None
                        continue
                    if not attended or offset[0] == offset[1] or not isinstance(label, int) or not 0 <= label < len(names):
                        issues["invalid_supervised_token"] += 1
                        active = None
                        continue
                    name = names[label]
                    if name == "O":
                        if active is not None:
                            issues["invalid_bioes_transition"] += 1
                        active = None
                        continue
                    prefix, category = name.split("-", 1)
                    if prefix in {"B", "S"}:
                        if active is not None:
                            issues["invalid_bioes_transition"] += 1
                        if prefix == "S":
                            emit(text, offsets, [position], category, gold)
                            active = None
                        else:
                            active = (category, [position])
                    elif prefix in {"I", "E"} and active is not None and active[0] == category:
                        active[1].append(position)
                        if prefix == "E":
                            emit(text, offsets, active[1], category, gold)
                            active = None
                    else:
                        issues["invalid_bioes_transition"] += 1
                        active = None
                if active is not None:
                    issues["invalid_bioes_transition"] += 1
    return {**counts, "issues": dict(issues), "passed": not bool(issues)}
