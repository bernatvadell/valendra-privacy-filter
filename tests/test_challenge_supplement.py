import pytest
import pyarrow.parquet as pq

from valendra_privacy_filter.datasets.challenge import prepare_address_challenge
from valendra_privacy_filter.datasets.mixed import audit_prepared_isolation, merge_prepared_datasets
from valendra_privacy_filter.datasets.nemotron import prepare_nemotron_records
from valendra_privacy_filter.datasets.prepare import prepare_records
from test_data import WindowTokenizer
from test_nemotron import POLICY_PATH


def test_address_challenge_is_deterministic_and_only_in_challenge(tmp_path):
    baseline = tmp_path / "baseline"
    prepare_records({}, WindowTokenizer(), baseline, max_length=128, overlap=16)
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare_address_challenge(baseline, a, WindowTokenizer(), count=20, seed=42)
    prepare_address_challenge(baseline, b, WindowTokenizer(), count=20, seed=42)
    assert first["splits"]["challenge"]["documents"] == 20
    assert first["splits"]["challenge"]["categories"]["ADDRESS"] == 20
    assert all(first["splits"][s]["documents"] == 0 for s in ("train", "validation", "test"))
    assert pq.read_table(a / "challenge.parquet").to_pylist() == pq.read_table(b / "challenge.parquet").to_pylist()


def test_mixed_dataset_accepts_independent_address_challenge_and_verifies_holdout(tmp_path):
    baseline, nvidia, supplement, mixed = (tmp_path / s for s in ("base", "nvidia", "supplement", "mixed"))
    prepare_records({}, WindowTokenizer(), baseline, max_length=128, overlap=16)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH, train_rows=[], test_rows=[])
    prepare_address_challenge(baseline, supplement, WindowTokenizer(), count=8, seed=42)
    result = merge_prepared_datasets(baseline, nvidia, mixed, allow_sampled_holdout=True,
                                     challenge_supplement_dir=supplement)
    assert result["splits"]["challenge"]["categories"]["ADDRESS"] == 8
    assert result["source_parameters"]["challenge_supplement_manifest_sha256"]
    assert audit_prepared_isolation(mixed, nvidia)["passed"]


def test_mixed_dataset_excludes_challenge_supplement_overlapping_baseline(tmp_path):
    import hashlib

    baseline, nvidia, supplement = (tmp_path / s for s in ("base", "nvidia", "supplement"))
    # Identical original text is sufficient to invalidate an independent supplement.
    row = {"source_text": "Example without PII.", "privacy_mask": [], "source": "synthetic",
           "template_id": "base", "language": "en", "region": "US"}
    prepare_records({"train": [row]}, WindowTokenizer(), baseline, max_length=128, overlap=16)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH, train_rows=[], test_rows=[])
    other = dict(row, template_id="different")
    baseline_hash = hashlib.sha256((baseline / "manifest.json").read_bytes()).hexdigest()
    prepare_records({"challenge": [other]}, WindowTokenizer(), supplement, max_length=128, overlap=16,
                    source_parameters={"baseline_manifest_sha256": baseline_hash})
    result = merge_prepared_datasets(baseline, nvidia, tmp_path / "mixed", allow_sampled_holdout=True,
                                     challenge_supplement_dir=supplement)
    assert result["splits"]["challenge"]["documents"] == 0
    assert result["leakage_checks"]["supplement_documents_excluded"] == 1
    assert result["postwrite_isolation_audit"]["passed"]


def test_challenge_drops_only_address_matching_other_source_not_entire_family(tmp_path):
    import hashlib

    baseline, nvidia, supplement = (tmp_path / s for s in ("base", "nvidia", "supplement"))

    def synthetic(text, address, template):
        start = text.index(address)
        return {"source_text": text, "privacy_mask": [{"start": start, "end": start + len(address),
                                                        "label": "ADDRESS", "value": address}],
                "source": "synthetic", "template_id": template, "language": "en", "region": "US"}

    shared, unique = "123 Example Street, Springfield", "456 Another Avenue, Hillside"
    prepare_records({"train": [synthetic(f"Residence: {shared}.", shared, "base")]},
                    WindowTokenizer(), baseline, max_length=128, overlap=16)
    prepare_nemotron_records(baseline, nvidia, WindowTokenizer(), POLICY_PATH, train_rows=[], test_rows=[])
    baseline_hash = hashlib.sha256((baseline / "manifest.json").read_bytes()).hexdigest()
    rows = [synthetic(f"Deliver to: {value}.", value, "challenge-address") for value in (shared, unique)]
    prepare_records({"challenge": rows}, WindowTokenizer(), supplement, max_length=128, overlap=16,
                    source_parameters={"baseline_manifest_sha256": baseline_hash})
    result = merge_prepared_datasets(baseline, nvidia, tmp_path / "mixed", allow_sampled_holdout=True,
                                     challenge_supplement_dir=supplement)
    assert result["splits"]["challenge"]["documents"] == 1
    assert result["leakage_checks"]["supplement_documents_excluded"] == 1
