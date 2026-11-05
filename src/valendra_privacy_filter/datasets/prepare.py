"""Streaming, grouped and auditable Parquet preparation."""

from collections import Counter
import hashlib
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from valendra_privacy_filter.datasets.alignment import align_labels
from valendra_privacy_filter.datasets.records import normalize_record, template_signature
from valendra_privacy_filter.labels import label_names

DATASET_ID = "ai4privacy/pii-masking-openpii-1m"
DATASET_REVISION = "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0"
SPLITS = ("train", "validation", "test", "challenge")
logger = logging.getLogger(__name__)
SPAN = pa.struct([("start", pa.int32()), ("end", pa.int32()), ("label", pa.string())])
RAW_SPAN = pa.struct([("start", pa.int32()), ("end", pa.int32()), ("label", pa.string()), ("value", pa.string())])
SCHEMA = pa.schema([
    ("document_id", pa.string()), ("group_id", pa.string()), ("source", pa.string()),
    ("language", pa.string()), ("region", pa.string()), ("text", pa.string()),
    ("spans", pa.list_(SPAN)), ("ignored", pa.list_(SPAN)), ("raw_annotations", pa.list_(RAW_SPAN)),
    ("window_index", pa.int32()), ("input_ids", pa.list_(pa.int64())),
    ("attention_mask", pa.list_(pa.int64())), ("labels", pa.list_(pa.int64())),
    ("offset_mapping", pa.list_(pa.list_(pa.int32()))),
])


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(dataset_dir, splits=SPLITS):
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for split in splits:
        if split not in SPLITS or split not in manifest.get("files", {}):
            raise ValueError("Dataset manifest is missing a split checksum")
        if _sha256_file(dataset_dir / f"{split}.parquet") != manifest["files"][split]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch for {split}")
    return manifest


def group_split(group_id, seed=42):
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def nemotron_group_split(group_id, seed=42):
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) % 10
    return "validation" if bucket == 0 else "train"


def stream_openpii(languages=("es", "en"), limit=None, spanish_regions=("ES",)):
    from datasets import load_dataset

    languages = tuple(dict.fromkeys(languages))
    if not languages or (limit is not None and limit < 1):
        raise ValueError("Provide languages and a positive limit")
    weights = {language: 2 if language == "es" else 1 for language in languages}
    quotas = None
    if limit is not None:
        total_weight = sum(weights.values())
        quotas = {language: limit * weights[language] // total_weight for language in languages}
        priority = sorted(languages, key=lambda lang: (-(limit * weights[lang] % total_weight), lang != "es", lang))
        for language in priority[:limit - sum(quotas.values())]:
            quotas[language] += 1
    selected = Counter()
    scanned = 0
    for original_split in ("train", "validation"):
        data = load_dataset(DATASET_ID, split=original_split, revision=DATASET_REVISION, streaming=True)
        for row in data:
            scanned += 1
            if scanned % 25_000 == 0:
                logger.info("OpenPII rows scanned=%s selected_by_language=%s", scanned, dict(selected))
            language = row.get("language")
            if language not in languages or (language == "es" and spanish_regions and row.get("region") not in spanish_regions):
                continue
            if quotas is not None and selected[language] >= quotas[language]:
                continue
            row["source"] = "openpii"
            yield row
            selected[language] += 1
            if quotas is not None and all(selected[language] >= quotas[language] for language in languages):
                return
    if quotas is not None and any(selected[language] < quotas[language] for language in languages):
        raise ValueError(f"OpenPII stream exhausted before language quotas were met: {dict(selected)}")


def prepare_records(sources, tokenizer, output_dir, max_length=512, overlap=64, seed=42,
                    categories=None, unknown_policy="ignore", source_parameters=None, source_metadata=None):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Preparation requires the Qwen3 fast tokenizer")
    if max_length < 2 or not 0 <= overlap < max_length:
        raise ValueError("Invalid window length or overlap")
    names = label_names() if categories is None else label_names(categories)
    allowed = {name.split("-", 1)[1] for name in names[1:]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writers = {split: pq.ParquetWriter(output / f"{split}.parquet", SCHEMA, compression="zstd") for split in SPLITS}
    buffers = {split: [] for split in SPLITS}
    counts = {split: {"documents": 0, "windows": 0, "languages": Counter(), "categories": Counter(),
                      "regions": Counter(), "sources": Counter()}
              for split in SPLITS}
    raw_labels, ignored_labels, reasons, alignment_issues = Counter(), Counter(), Counter(), Counter()
    seen, group_owners, duplicates = {}, {}, 0

    def flush(split):
        if buffers[split]:
            writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=SCHEMA))
            buffers[split].clear()

    try:
        for source_split, rows in sources.items():
            if source_split not in {*SPLITS, "openpii", "nemotron_train"}:
                raise ValueError(f"Unknown source split: {source_split}")
            for row in rows:
                try:
                    record = normalize_record(row, unknown_policy=unknown_policy)
                    if record["source"] == "nemotron":
                        from valendra_privacy_filter.datasets.mixed import canonical_signature
                        group = "nemotron:" + canonical_signature(record["text"], record["spans"], record["ignored"])
                    else:
                        group = ("synthetic:" + str(row["template_id"]) if row.get("source") == "synthetic"
                                 else "openpii:" + template_signature(row))
                    if source_split == "openpii":
                        split = group_split(group, seed)
                    elif source_split == "nemotron_train":
                        split = nemotron_group_split(group, seed)
                    else:
                        split = source_split
                    if group in group_owners and group_owners[group] != split:
                        raise ValueError("template group assigned to multiple splits")
                    group_owners[group] = split
                    text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                    canonical_annotations = [{key: item[key] for key in ("start", "end", "label", "value")}
                                             for item in record["raw_annotations"]]
                    annotation_hash = hashlib.sha256(json.dumps(canonical_annotations, sort_keys=True).encode()).hexdigest()
                    if text_hash in seen:
                        duplicates += 1
                        if seen[text_hash] != annotation_hash:
                            raise ValueError("duplicate text has conflicting annotations")
                        continue
                    seen[text_hash] = annotation_hash
                    if not record["text"]:
                        reasons["empty_text"] += 1
                        continue
                    for span in record["spans"][:]:
                        if span["label"] not in allowed:
                            record["spans"].remove(span)
                            record["ignored"].append(span)
                    encoded = tokenizer(record["text"], max_length=max_length, stride=overlap,
                                        truncation=True, return_overflowing_tokens=True,
                                        return_offsets_mapping=True, add_special_tokens=False)
                    counts[split]["documents"] += 1
                    if sum(value["documents"] for value in counts.values()) % 10_000 == 0:
                        logger.info("Prepared documents by split: %s", {key: value["documents"] for key, value in counts.items()})
                    counts[split]["languages"][record["language"]] += 1
                    counts[split]["regions"][record["region"]] += 1
                    counts[split]["sources"][record["source"]] += 1
                    counts[split]["categories"].update(span["label"] for span in record["spans"])
                    raw_labels.update(item["label"] for item in record["raw_annotations"])
                    ignored_labels.update(item["label"] for item in record["ignored"])
                    for index, ids in enumerate(encoded["input_ids"]):
                        offsets = encoded["offset_mapping"][index]
                        labels, issues = align_labels(record["text"], offsets, record["spans"], record["ignored"], names,
                                                      return_diagnostics=True)
                        alignment_issues.update(issues)
                        raw_annotations = [{key: item[key] for key in ("start", "end", "label", "value")}
                                           for item in record["raw_annotations"]]
                        buffers[split].append({"document_id": text_hash, "group_id": group,
                            "source": record["source"], "language": record["language"], "region": record["region"],
                            "text": record["text"], "spans": record["spans"], "ignored": record["ignored"],
                            "raw_annotations": raw_annotations, "window_index": index, "input_ids": ids,
                            "attention_mask": encoded["attention_mask"][index], "labels": labels,
                            "offset_mapping": offsets})
                        counts[split]["windows"] += 1
                        if len(buffers[split]) >= 512:
                            flush(split)
                except (ValueError, KeyError, TypeError) as exc:
                    if unknown_policy == "error" or "multiple splits" in str(exc) or "conflicting annotations" in str(exc):
                        raise
                    reasons[type(exc).__name__] += 1
        for split in SPLITS:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()
    manifest = {
        "sources": source_metadata if source_metadata is not None else [
            {"id": DATASET_ID, "revision": DATASET_REVISION, "license": "CC-BY-4.0",
             "attribution": "Ai4Privacy / Ai Suisse SA"},
            {"id": "local-deterministic-synthetic", "license": "Apache-2.0"}],
        "tokenizer": {"id": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"},
        "seed": seed, "max_length": max_length, "overlap": overlap, "label_names": names,
        "source_parameters": source_parameters or {},
        "unknown_policy": unknown_policy, "raw_label_counts": dict(raw_labels),
        "ignored_label_counts": dict(ignored_labels), "quarantine_reasons": dict(reasons),
        "alignment_issues_by_window": dict(alignment_issues),
        "duplicates_removed": duplicates, "group_cross_split_count": 0, "splits": counts,
        "files": {split: {"sha256": _sha256_file(output / f"{split}.parquet"),
                          "bytes": (output / f"{split}.parquet").stat().st_size} for split in SPLITS},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
