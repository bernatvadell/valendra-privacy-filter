"""Cross-source grouping and holdout protection for Nemotron-PII."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from valendra_privacy_filter.datasets.nemotron import AnnotationError, adapt_nemotron_record
from valendra_privacy_filter.datasets.prepare import SCHEMA, SPLITS, _sha256_file, nemotron_group_split, verify_dataset
from valendra_privacy_filter.datasets.records import normalize_record

IDENTITY_CATEGORIES = frozenset({"EMAIL", "PHONE", "GOVERNMENT_ID", "FINANCIAL_ID", "SECRET", "ADDRESS"})
IDENTITY_MIN_LENGTH = 8
ADDRESS_MIN_LENGTH = 15


def identity_fingerprints(text, spans):
    """Hash high-signal annotated values; hashes remain in memory, never in manifests."""
    result = set()
    for item in spans:
        category = item["label"]
        if category not in IDENTITY_CATEGORIES:
            continue
        start, end = item["start"], item["end"]
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid identity fingerprint bounds")
        value = unicodedata.normalize("NFKC", text[start:end]).casefold().strip()
        if category == "ADDRESS":
            value = re.sub(r"\s+", " ", value)
            if len(value) < ADDRESS_MIN_LENGTH or not any(character.isdigit() for character in value):
                continue
        if category in {"PHONE", "GOVERNMENT_ID", "FINANCIAL_ID"}:
            value = re.sub(r"[^a-z0-9]", "", value)
        if len(value) >= IDENTITY_MIN_LENGTH:
            result.add((category, hashlib.sha256(value.encode("utf-8")).digest()))
    return result


def canonical_signature(text, spans, ignored):
    """Hash a normalized template using project categories, not source labels."""
    entries = [(item["start"], item["end"], item["label"]) for item in spans]
    entries += [(item["start"], item["end"], "IGNORED") for item in ignored]
    cursor, parts = 0, []
    for start, end, category in sorted(entries, key=lambda item: (item[0], -item[1], item[2])):
        if start < cursor:
            continue
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid canonical template bounds")
        parts.extend((text[cursor:start], f"<{category}>"))
        cursor = end
    parts.append(text[cursor:])
    normalized = re.sub(r"\s+", " ", "".join(parts)).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class BaselineIndex:
    def __init__(self, manifest_sha256):
        self.manifest_sha256 = manifest_sha256
        self.document_ids = set()
        self.document_owners = {}
        self.conflicting_documents = set()
        self.group_owners = {}
        self.conflicting_groups = set()
        self.tainted_groups = set()
        self.signatures = set()
        self.signature_owners = {}
        self.conflicting_signatures = set()
        self.identity_owners = {}
        self.conflicting_identities = set()
        self.identity_tainted_groups = set()
        self.identity_tainted_documents = set()

    @classmethod
    def from_directory(cls, directory):
        directory = Path(directory)
        verify_dataset(directory)
        index = cls(hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest())
        columns = ["document_id", "group_id", "text", "spans", "ignored", "window_index"]
        for split in SPLITS:
            for batch in pq.ParquetFile(directory / f"{split}.parquet").iter_batches(columns=columns):
                for row in batch.to_pylist():
                    if row["window_index"] != 0:
                        continue
                    if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["document_id"]:
                        raise ValueError("Baseline dataset document hash mismatch")
                    index.document_ids.add(row["document_id"])
                    previous_document = index.document_owners.get(row["document_id"])
                    if previous_document is None:
                        index.document_owners[row["document_id"]] = split
                    elif previous_document != split:
                        index.conflicting_documents.add(row["document_id"])
                    previous_group = index.group_owners.get(row["group_id"])
                    if previous_group is None:
                        index.group_owners[row["group_id"]] = split
                    elif previous_group != split:
                        index.conflicting_groups.add(row["group_id"])
                    signature = canonical_signature(row["text"], row["spans"], row["ignored"])
                    index.signatures.add(signature)
                    previous = index.signature_owners.get(signature)
                    if previous is None:
                        index.signature_owners[signature] = split
                    elif previous != split:
                        index.conflicting_signatures.add(signature)
                    for fingerprint in identity_fingerprints(row["text"], row["spans"]):
                        owner = index.identity_owners.setdefault(fingerprint, split)
                        if owner != split:
                            index.conflicting_identities.add(fingerprint)
        index.tainted_groups.update(index.conflicting_groups)
        if index.conflicting_signatures or index.conflicting_documents or index.conflicting_identities:
            for split in SPLITS:
                for batch in pq.ParquetFile(directory / f"{split}.parquet").iter_batches(columns=columns):
                    for row in batch.to_pylist():
                        if row["window_index"] != 0:
                            continue
                        signature = canonical_signature(row["text"], row["spans"], row["ignored"])
                        if signature in index.conflicting_signatures or row["document_id"] in index.conflicting_documents:
                            index.tainted_groups.add(row["group_id"])
                        fingerprints = identity_fingerprints(row["text"], row["spans"])
                        if any(index.identity_owners.get(key) != split for key in fingerprints):
                            index.identity_tainted_groups.add(row["group_id"])
                            index.identity_tainted_documents.add(row["document_id"])
        return index


class NemotronFilter:
    """Exclude baseline overlaps and reserve NVIDIA test before train is read."""

    def __init__(self, baseline, policy, seed=42):
        self.baseline, self.policy, self.seed = baseline, policy, seed
        self.stats = {"train": Counter(), "test": Counter()}
        self.source_labels = {"train": Counter(), "test": Counter()}
        self.unknown_labels = {"train": Counter(), "test": Counter()}
        self._test_ids, self._test_signatures = set(), set()
        self._test_identities, self._train_identity_owners = set(), {}
        self._test_finished = False

    def rows(self, split, source_rows):
        if split not in {"train", "test"}:
            raise ValueError("NVIDIA source split must be train or test")
        if split == "train" and not self._test_finished:
            raise ValueError("Reserve the complete NVIDIA test split first")
        for raw in source_rows:
            self.stats[split]["source_records"] += 1
            try:
                converted = adapt_nemotron_record(raw, self.policy)
                normalized = normalize_record(converted)
            except (AnnotationError, ValueError, TypeError, KeyError) as exc:
                reason = str(exc) if isinstance(exc, AnnotationError) else "invalid_record"
                self.stats[split]["quarantine_" + reason] += 1
                continue
            for annotation in converted["privacy_mask"]:
                self.source_labels[split][annotation["label"]] += 1
                if annotation["mapping_status"] == "unknown":
                    self.unknown_labels[split][annotation["label"]] += 1
            text_id = hashlib.sha256(normalized["text"].encode("utf-8")).hexdigest()
            signature = canonical_signature(normalized["text"], normalized["spans"], normalized["ignored"])
            identities = identity_fingerprints(normalized["text"], normalized["spans"])
            if text_id in self.baseline.document_ids:
                self.stats[split]["baseline_exact_overlap"] += 1
                continue
            if signature in self.baseline.signatures:
                self.stats[split]["baseline_template_overlap"] += 1
                continue
            if any(key in self.baseline.identity_owners for key in identities):
                self.stats[split]["baseline_identity_overlap"] += 1
                continue
            if split == "train" and text_id in self._test_ids:
                self.stats[split]["nvidia_test_exact_overlap"] += 1
                continue
            if split == "train" and signature in self._test_signatures:
                self.stats[split]["nvidia_test_template_overlap"] += 1
                continue
            if split == "train" and identities & self._test_identities:
                self.stats[split]["nvidia_test_identity_overlap"] += 1
                continue
            if split == "train":
                assigned = nemotron_group_split("nemotron:" + signature, self.seed)
                if any(key in self._train_identity_owners and self._train_identity_owners[key] != assigned
                       for key in identities):
                    self.stats[split]["nvidia_cross_split_identity_overlap"] += 1
                    continue
                for key in identities:
                    self._train_identity_owners[key] = assigned
            if split == "test":
                self._test_ids.add(text_id)
                self._test_signatures.add(signature)
                self._test_identities.update(identities)
            self.stats[split]["accepted_records"] += 1
            yield converted
        if split == "test":
            self._test_finished = True


def _check_nemotron_isolation(directory, baseline):
    owners_by_id, owners_by_signature, owners_by_group, owners_by_identity = {}, {}, {}, {}
    columns = ["document_id", "group_id", "text", "spans", "ignored", "window_index", "source"]
    for split in SPLITS:
        for batch in pq.ParquetFile(directory / f"{split}.parquet").iter_batches(columns=columns):
            for row in batch.to_pylist():
                if row["window_index"] != 0:
                    continue
                if row["source"] != "nemotron":
                    raise ValueError("Additional dataset contains an unexpected source")
                text_id = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if text_id != row["document_id"]:
                    raise ValueError("Additional dataset document hash mismatch")
                signature = canonical_signature(row["text"], row["spans"], row["ignored"])
                if text_id in baseline.document_ids or signature in baseline.signatures:
                    raise ValueError("Additional dataset overlaps the baseline")
                fingerprints = identity_fingerprints(row["text"], row["spans"])
                if any(key in baseline.identity_owners for key in fingerprints):
                    raise ValueError("Additional dataset shares identifiers with the baseline")
                if (owners_by_id.get(text_id, split) != split
                        or owners_by_signature.get(signature, split) != split
                        or owners_by_group.get(row["group_id"], split) != split
                        or any(owners_by_identity.get(key, split) != split for key in fingerprints)):
                    raise ValueError("Additional dataset contains cross-split overlap")
                owners_by_id[text_id] = split
                owners_by_signature[signature] = split
                owners_by_group[row["group_id"]] = split
                for key in fingerprints:
                    owners_by_identity[key] = split
    if any(owner == "challenge" for owner in owners_by_id.values()):
        raise ValueError("NVIDIA challenge records are not supported")


def audit_prepared_isolation(dataset_dir, holdout_dir=None):
    """Re-read produced Parquet and report cross-split and optional holdout overlaps."""
    index = BaselineIndex.from_directory(dataset_dir)
    report = {"cross_split_exact_texts": len(index.conflicting_documents),
              "cross_split_templates": len(index.conflicting_signatures),
              "cross_split_groups": len(index.conflicting_groups),
              "cross_split_identity_values": len(index.conflicting_identities),
              "holdout_documents": 0, "holdout_exact_matches": 0,
              "holdout_template_matches": 0, "holdout_identity_matches": 0,
              "holdout_overlap_documents": 0}
    if holdout_dir is not None:
        holdout_dir = Path(holdout_dir)
        verify_dataset(holdout_dir, ["test"])
        columns = ["document_id", "text", "spans", "ignored", "window_index"]
        for batch in pq.ParquetFile(holdout_dir / "test.parquet").iter_batches(columns=columns):
            for row in batch.to_pylist():
                if row["window_index"] != 0:
                    continue
                text_id = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if text_id != row["document_id"]:
                    raise ValueError("Holdout dataset document hash mismatch")
                signature = canonical_signature(row["text"], row["spans"], row["ignored"])
                exact, template = text_id in index.document_ids, signature in index.signatures
                identity = any(key in index.identity_owners for key in identity_fingerprints(row["text"], row["spans"]))
                report["holdout_documents"] += 1
                report["holdout_exact_matches"] += int(exact)
                report["holdout_template_matches"] += int(template)
                report["holdout_identity_matches"] += int(identity)
                report["holdout_overlap_documents"] += int(exact or template or identity)
    report["passed"] = not any(report[key] for key in (
        "cross_split_exact_texts", "cross_split_templates", "cross_split_groups",
        "cross_split_identity_values", "holdout_overlap_documents"))
    return report


def merge_prepared_datasets(baseline_dir, nvidia_dir, output_dir, allow_sampled_holdout=False,
                            challenge_supplement_dir=None):
    """Make an isolated mixed copy; preserve NVIDIA test in its own dataset."""
    baseline_dir, nvidia_dir, output_dir = map(lambda p: Path(p).resolve(), (baseline_dir, nvidia_dir, output_dir))
    if any(output_dir == parent or parent in output_dir.parents for parent in (baseline_dir, nvidia_dir)):
        raise ValueError("Output must be separate from both input datasets")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    baseline_manifest = verify_dataset(baseline_dir)
    nvidia_manifest = verify_dataset(nvidia_dir)
    index = BaselineIndex.from_directory(baseline_dir)
    params = nvidia_manifest.get("source_parameters", {})
    if params.get("baseline_manifest_sha256") != index.manifest_sha256:
        raise ValueError("NVIDIA dataset was not filtered against this baseline")
    if not params.get("nvidia_test_is_complete", False) and not allow_sampled_holdout:
        raise ValueError("NVIDIA test is a sample; explicitly allow a smoke-only mixed dataset")
    for key in ("label_names", "tokenizer", "max_length", "overlap"):
        if baseline_manifest.get(key) != nvidia_manifest.get(key):
            raise ValueError(f"Dataset {key} mismatch")
    _check_nemotron_isolation(nvidia_dir, index)
    supplement_dir = Path(challenge_supplement_dir).resolve() if challenge_supplement_dir else None
    supplement_manifest = None
    supplement_rejected_documents, supplement_rejected_groups = set(), set()
    supplement_overlap_reasons = Counter()
    if supplement_dir is not None:
        if output_dir == supplement_dir or supplement_dir in output_dir.parents:
            raise ValueError("Output must be separate from the challenge supplement")
        supplement_manifest = verify_dataset(supplement_dir)
        if supplement_manifest.get("source_parameters", {}).get("baseline_manifest_sha256") != index.manifest_sha256:
            raise ValueError("Challenge supplement was prepared for a different baseline")
        for key in ("label_names", "tokenizer", "max_length", "overlap"):
            if baseline_manifest.get(key) != supplement_manifest.get(key):
                raise ValueError(f"Challenge supplement {key} mismatch")
        if any(supplement_manifest["splits"][split]["documents"] for split in ("train", "validation", "test")):
            raise ValueError("Challenge supplement must contain only challenge records")
        supplement_index = BaselineIndex.from_directory(supplement_dir)
        nvidia_index = BaselineIndex.from_directory(nvidia_dir)
        references = (index, nvidia_index)
        columns = ["document_id", "group_id", "text", "spans", "ignored", "window_index", "source"]
        for batch in pq.ParquetFile(supplement_dir / "challenge.parquet").iter_batches(columns=columns):
            for row in batch.to_pylist():
                if row["window_index"] != 0:
                    continue
                if row["source"] != "synthetic":
                    raise ValueError("Challenge supplement must contain synthetic records")
                signature = canonical_signature(row["text"], row["spans"], row["ignored"])
                identities = identity_fingerprints(row["text"], row["spans"])
                if any(signature in ref.signatures or row["group_id"] in ref.group_owners for ref in references):
                    supplement_rejected_groups.add(row["group_id"])
                    supplement_overlap_reasons["template_or_group"] += 1
                if any(row["document_id"] in ref.document_ids for ref in references):
                    supplement_rejected_documents.add(row["document_id"])
                    supplement_overlap_reasons["exact_text"] += 1
                if any(any(key in ref.identity_owners for key in identities) for ref in references):
                    supplement_rejected_documents.add(row["document_id"])
                    supplement_overlap_reasons["identity_value"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {split: {"documents": 0, "windows": 0, "languages": Counter(), "categories": Counter(),
                      "regions": Counter(), "sources": Counter()} for split in SPLITS}
    raw_labels, ignored_labels = Counter(), Counter()
    excluded_documents = excluded_windows = 0
    supplement_excluded_documents = supplement_excluded_windows = 0
    excluded_identity_documents = Counter()
    for split in SPLITS:
        with pq.ParquetWriter(output_dir / f"{split}.parquet", SCHEMA, compression="zstd") as writer:
            directories = [baseline_dir] + ([nvidia_dir] if split in {"train", "validation"} else [])
            if split == "challenge" and supplement_dir is not None:
                directories.append(supplement_dir)
            for source_dir in directories:
                for batch in pq.ParquetFile(source_dir / f"{split}.parquet").iter_batches(batch_size=512):
                    included = []
                    for row in batch.to_pylist():
                        signature = canonical_signature(row["text"], row["spans"], row["ignored"])
                        if source_dir == supplement_dir and (row["document_id"] in supplement_rejected_documents
                                                             or row["group_id"] in supplement_rejected_groups):
                            supplement_excluded_windows += 1
                            supplement_excluded_documents += int(row["window_index"] == 0)
                            continue
                        if source_dir == baseline_dir and (signature in index.conflicting_signatures
                                                           or row["document_id"] in index.conflicting_documents
                                                           or row["group_id"] in index.tainted_groups
                                                           or row["document_id"] in index.identity_tainted_documents):
                            excluded_windows += 1
                            excluded_documents += int(row["window_index"] == 0)
                            if row["window_index"] == 0 and row["document_id"] in index.identity_tainted_documents:
                                excluded_identity_documents[split] += 1
                            continue
                        included.append(row)
                        counts[split]["windows"] += 1
                        if row["window_index"] == 0:
                            counts[split]["documents"] += 1
                            counts[split]["languages"][row["language"]] += 1
                            counts[split]["regions"][row["region"]] += 1
                            counts[split]["sources"][row["source"]] += 1
                            counts[split]["categories"].update(item["label"] for item in row["spans"])
                            raw_labels.update(item["label"] for item in row["raw_annotations"])
                            ignored_labels.update(item["label"] for item in row["ignored"])
                    if included:
                        writer.write_table(pa.Table.from_pylist(included, schema=SCHEMA))

    seen_sources, sources = set(), []
    for source in baseline_manifest["sources"] + nvidia_manifest["sources"] + (supplement_manifest["sources"] if supplement_manifest else []):
        if source["id"] not in seen_sources:
            sources.append(source)
            seen_sources.add(source["id"])
    manifest = {
        "sources": sources, "tokenizer": baseline_manifest["tokenizer"], "seed": baseline_manifest["seed"],
        "max_length": baseline_manifest["max_length"], "overlap": baseline_manifest["overlap"],
        "label_names": baseline_manifest["label_names"], "unknown_policy": "ignore",
        "source_parameters": {"baseline_manifest_sha256": index.manifest_sha256,
                              "nvidia_manifest_sha256": _sha256_file(nvidia_dir / "manifest.json"),
                              "nvidia_test_in_mixed": False,
                              "sampled_nvidia_holdout": not params.get("nvidia_test_is_complete", False),
                              "nvidia_train_is_complete": params.get("nvidia_train_is_complete", False),
                              "challenge_supplement_manifest_sha256": (_sha256_file(supplement_dir / "manifest.json")
                                                                       if supplement_dir else None),
                              "identity_categories": sorted(IDENTITY_CATEGORIES),
                              "identity_min_normalized_length": IDENTITY_MIN_LENGTH,
                              "identity_address_min_length": ADDRESS_MIN_LENGTH,
                              "identity_address_requires_digit": True,
                              "identity_split_priority": list(SPLITS)},
        "leakage_checks": {"baseline_cross_split_templates": len(index.conflicting_signatures),
                           "baseline_cross_split_exact_texts": len(index.conflicting_documents),
                           "baseline_cross_split_groups": len(index.conflicting_groups),
                           "baseline_cross_split_identity_values": len(index.conflicting_identities),
                           "baseline_groups_excluded": len(index.tainted_groups),
                           "baseline_identity_groups_affected": len(index.identity_tainted_groups),
                           "baseline_identity_documents_excluded_by_split": dict(excluded_identity_documents),
                           "baseline_documents_excluded": excluded_documents,
                           "baseline_windows_excluded": excluded_windows,
                           "supplement_documents_excluded": supplement_excluded_documents,
                           "supplement_windows_excluded": supplement_excluded_windows,
                           "supplement_overlap_reasons": dict(supplement_overlap_reasons),
                           "cross_source_overlap": 0, "nvidia_cross_split_overlap": 0},
        "raw_label_counts": dict(raw_labels), "ignored_label_counts": dict(ignored_labels),
        "component_alignment_issues_by_window": {
            "baseline": baseline_manifest.get("alignment_issues_by_window", {}),
            "nvidia": nvidia_manifest.get("alignment_issues_by_window", {}),
        },
        "component_alignment_issues_scope": "original component datasets before exclusions and holdout removal",
        "group_cross_split_count": 0, "splits": counts,
        "files": {split: {"sha256": _sha256_file(output_dir / f"{split}.parquet"),
                          "bytes": (output_dir / f"{split}.parquet").stat().st_size} for split in SPLITS},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = audit_prepared_isolation(output_dir, nvidia_dir)
    if not audit["passed"]:
        manifest_path.unlink()
        raise ValueError("Post-write dataset isolation audit failed")
    manifest["postwrite_isolation_audit"] = audit
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
