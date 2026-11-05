"""Evaluate prepared document-level splits and compare compatible experiments."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from valendra_privacy_filter.evaluation.metrics import evaluate_documents
from valendra_privacy_filter.datasets.prepare import verify_dataset


def evaluate_split(detector, dataset_dir, split, checkpoint=None):
    if split not in {"train", "validation", "test", "challenge"}:
        raise ValueError("Unsupported split")
    dataset_dir = Path(dataset_dir)
    verify_dataset(dataset_dir, [split])
    manifest = (dataset_dir / "manifest.json").read_bytes()
    seen = set()

    def documents():
        columns = ["document_id", "text", "language", "spans", "ignored"]
        for batch in pq.ParquetFile(dataset_dir / f"{split}.parquet").iter_batches(columns=columns):
            for row in batch.to_pylist():
                if row["document_id"] in seen:
                    continue
                seen.add(row["document_id"])
                yield {"language": row["language"], "gold": row["spans"], "ignored": row["ignored"],
                       "predicted": detector.detect(row["text"])["entities"]}

    result = evaluate_documents(documents())
    result["metadata"] = {"split": split, "dataset_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                          "attention_mode": getattr(detector.model.config, "privacy_attention_mode", "unknown"),
                          "max_length": detector.max_length, "overlap": detector.overlap,
                          "temperature": detector.temperature}
    if checkpoint is not None:
        identity_path = Path(checkpoint).parent / "run_identity.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            result["metadata"]["training_dataset_manifest_sha256"] = identity["dataset_manifest_sha256"]
            result["metadata"]["seed"] = identity["seed"]
            result["metadata"]["model_revision"] = identity["model_revision"]
            protocol = getattr(detector.model.config, "privacy_continuation_protocol_sha256", None)
            result["metadata"]["protocol_sha256"] = protocol or identity["protocol_sha256"]
            if protocol:
                result["metadata"]["base_protocol_sha256"] = identity["protocol_sha256"]
    return result


def compare_reports(first, second):
    a, b = first["metadata"], second["metadata"]
    keys = ("split", "dataset_manifest_sha256", "training_dataset_manifest_sha256", "seed", "model_revision",
            "max_length", "overlap", "protocol_sha256")
    if any(key not in a or key not in b or a[key] != b[key] for key in keys):
        raise ValueError("Reports have incompatible or missing dataset/protocol metadata")
    if {a["attention_mode"], b["attention_mode"]} != {"causal", "bidirectional"}:
        raise ValueError("Expected one causal and one bidirectional result")
    return {a["attention_mode"]: first, b["attention_mode"]: second,
            "dataset_manifest_sha256": a["dataset_manifest_sha256"], "split": a["split"]}
