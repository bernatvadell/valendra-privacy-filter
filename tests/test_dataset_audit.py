import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from valendra_privacy_filter.datasets.audit import audit_prepared_annotations
from valendra_privacy_filter.datasets.prepare import prepare_records
from valendra_privacy_filter.cli import app
from test_data import WindowTokenizer


def test_audit_detects_invalid_bioes_even_with_valid_file_checksum(tmp_path):
    text = "Ana García llegó."
    row = {"source_text": text, "source": "openpii", "language": "es", "region": "ES", "privacy_mask": [
        {"start": 0, "end": 10, "label": "PERSON", "value": "Ana García"}]}
    prepare_records({"train": [row]}, WindowTokenizer(), tmp_path, max_length=64, overlap=8)
    first = audit_prepared_annotations(tmp_path)
    assert first["passed"] and first["entity_spans_checked"] == 1

    path = tmp_path / "train.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["labels"][0] = 2  # I-PERSON cannot begin a supervised sequence.
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = audit_prepared_annotations(tmp_path)
    assert not result["passed"]
    assert result["issues"]["invalid_bioes_transition"] > 0


def test_audit_detects_raw_span_value_mismatch(tmp_path):
    text = "Contact ana@example.com."
    start = text.index("ana@example.com")
    row = {"source_text": text, "source": "openpii", "language": "en", "region": "US", "privacy_mask": [
        {"start": start, "end": start + len("ana@example.com"), "label": "EMAIL", "value": "ana@example.com"}]}
    prepare_records({"train": [row]}, WindowTokenizer(), tmp_path, max_length=64, overlap=8)
    path = tmp_path / "train.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["raw_annotations"][0]["value"] = "not-the-original"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = audit_prepared_annotations(tmp_path)
    assert not result["passed"]
    assert result["issues"]["raw_span_value_mismatch"] == 1


def test_cli_can_save_aggregate_annotation_audit(tmp_path):
    from typer.testing import CliRunner

    data, report = tmp_path / "data", tmp_path / "audit.json"
    prepare_records({}, WindowTokenizer(), data, max_length=64, overlap=8)
    result = CliRunner().invoke(app, ["audit-dataset-annotations", "--data-dir", str(data), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text())["passed"]
