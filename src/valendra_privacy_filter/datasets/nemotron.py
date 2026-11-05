"""Strict adapter for the published, pre-generated NVIDIA Nemotron-PII dataset."""

import ast
import hashlib
import json
from importlib.resources import files
import math
from pathlib import Path

import yaml

from valendra_privacy_filter.labels import DEFAULT_CATEGORIES

DATASET_ID = "nvidia/Nemotron-PII"
DATASET_REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
MAX_LITERAL_LENGTH = 250_000


class AnnotationError(ValueError):
    """Sanitized annotation failure; its message never contains source text."""


def _label_policy_bytes(path):
    if path is None:
        return files("valendra_privacy_filter").joinpath("resources/nemotron-labels.yaml").read_bytes()
    return Path(path).read_bytes()


def load_label_policy(path=None):
    value = yaml.safe_load(_label_policy_bytes(path).decode("utf-8"))
    if not isinstance(value, dict) or value.get("unknown") != "ignore":
        raise ValueError("Policy must explicitly set unknown: ignore")
    mapping, ignored = value.get("mapping"), value.get("ignore", [])
    if not isinstance(mapping, dict) or not isinstance(ignored, list):
        raise ValueError("Policy mapping and ignore must be a mapping and list")
    if any(not isinstance(key, str) or not key for key in mapping) or any(not isinstance(key, str) or not key for key in ignored):
        raise ValueError("Source labels must be nonempty strings")
    if any(category not in DEFAULT_CATEGORIES for category in mapping.values()):
        raise ValueError("Policy contains an unsupported project category")
    if set(mapping) & set(ignored) or len(set(ignored)) != len(ignored):
        raise ValueError("A source label cannot be both mapped and ignored")
    return {"mapping": dict(mapping), "ignore": frozenset(ignored), "unknown": "ignore"}


def adapt_nemotron_record(row, policy):
    text, spans = row.get("text"), row.get("spans")
    if not isinstance(text, str):
        raise AnnotationError("invalid_source_text")
    if isinstance(spans, str):
        if len(spans) > MAX_LITERAL_LENGTH:
            raise AnnotationError("oversized_spans_literal")
        try:
            spans = ast.literal_eval(spans)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            raise AnnotationError("invalid_spans_literal") from None
    if not isinstance(spans, (list, tuple)):
        raise AnnotationError("invalid_spans_list")
    converted = []
    for span in spans:
        if not isinstance(span, dict):
            raise AnnotationError("invalid_span")
        start, end, label = span.get("start"), span.get("end"), span.get("label")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise AnnotationError("invalid_span_bounds")
        if not isinstance(label, str) or not label:
            raise AnnotationError("invalid_span_label")
        value = span.get("text")
        if type(value) in (int, float):
            if isinstance(value, float) and not math.isfinite(value):
                raise AnnotationError("invalid_span_value")
            value = str(value)
        if not isinstance(value, str):
            raise AnnotationError("invalid_span_value")
        if text[start:end] != value:
            raise AnnotationError("span_value_mismatch")
        mapped = policy["mapping"].get(label)
        status = "mapped" if mapped else "ignored" if label in policy["ignore"] else "unknown"
        converted.append({"start": start, "end": end, "label": label, "value": value,
                          "normalized_label": mapped, "mapping_status": status})
    locale = row.get("locale")
    return {"source_text": text, "privacy_mask": converted, "language": "en",
            "region": {"us": "US", "intl": "INTL"}.get(locale, "UNKNOWN"),
            "source": "nemotron", "uid": str(row.get("uid", ""))}


def stream_nemotron(split, limit=None):
    """Read the pinned original Parquet file; limit counts source rows, not valid rows."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if split not in {"train", "test"}:
        raise ValueError("NVIDIA split must be train or test")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Source limit must be a positive integer")
    path = hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
                           filename=f"data/{split}-00000-of-00001.parquet")
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["uid", "locale", "text", "spans"]):
        for row in batch.to_pylist():
            yield row
            count += 1
            if limit is not None and count >= limit:
                return


def prepare_nemotron_records(baseline_dir, output_dir, tokenizer, policy_path, *, train_rows, test_rows,
                             seed=42, train_is_complete=False, test_is_complete=False):
    """Prepare NVIDIA independently, excluding baseline and official-test overlaps."""
    from valendra_privacy_filter.datasets.mixed import (BaselineIndex, NemotronFilter, IDENTITY_CATEGORIES,
                                                    IDENTITY_MIN_LENGTH, ADDRESS_MIN_LENGTH, _check_nemotron_isolation,
                                                    audit_prepared_isolation)
    from valendra_privacy_filter.datasets.prepare import prepare_records, verify_dataset

    baseline_dir, output_dir = Path(baseline_dir).resolve(), Path(output_dir).resolve()
    if output_dir == baseline_dir or baseline_dir in output_dir.parents:
        raise ValueError("Output must be separate from the baseline dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    baseline = verify_dataset(baseline_dir)
    index = BaselineIndex.from_directory(baseline_dir)
    policy = load_label_policy(policy_path)
    filtered = NemotronFilter(index, policy, seed=seed)
    categories = [name[2:] for name in baseline["label_names"] if name.startswith("B-")]
    policy_sha = hashlib.sha256(_label_policy_bytes(policy_path)).hexdigest()
    metadata = {"baseline_manifest_sha256": index.manifest_sha256, "label_policy_sha256": policy_sha,
                "label_mapping": policy["mapping"], "ignored_source_labels": sorted(policy["ignore"]),
                "unknown_source_label_policy": policy["unknown"],
                "nvidia_train_is_complete": train_is_complete, "nvidia_test_is_complete": test_is_complete,
                "nvidia_split_policy": "official test reserved; train grouped into 90/10 train/validation",
                "identity_categories": sorted(IDENTITY_CATEGORIES), "identity_min_normalized_length": IDENTITY_MIN_LENGTH,
                "identity_address_min_length": ADDRESS_MIN_LENGTH, "identity_address_requires_digit": True,
                "baseline_conflicting_template_count": len(index.conflicting_signatures),
                "baseline_tainted_group_count": len(index.tainted_groups),
                "filter_statistics": filtered.stats, "source_label_counts": filtered.source_labels,
                "unknown_label_counts": filtered.unknown_labels}
    manifest = prepare_records(
        {"test": filtered.rows("test", test_rows), "nemotron_train": filtered.rows("train", train_rows)},
        tokenizer, output_dir, max_length=baseline["max_length"], overlap=baseline["overlap"],
        seed=seed, categories=categories, source_parameters=metadata,
        source_metadata=[{"id": DATASET_ID, "revision": DATASET_REVISION, "license": "CC-BY-4.0",
                          "attribution": "NVIDIA Corporation"}],
    )
    manifest_path = output_dir / "manifest.json"
    try:
        _check_nemotron_isolation(output_dir, index)
        audit = audit_prepared_isolation(output_dir)
        if not audit["passed"]:
            raise ValueError("Post-write NVIDIA split isolation audit failed")
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    manifest["postwrite_isolation_audit"] = audit
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
