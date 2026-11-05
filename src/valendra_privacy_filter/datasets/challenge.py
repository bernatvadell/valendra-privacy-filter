"""Independent synthetic postal-address challenge, never used for training."""

import hashlib
from pathlib import Path

from faker import Faker

from valendra_privacy_filter.datasets.prepare import prepare_records, verify_dataset


def address_challenge_rows(count, seed):
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    fakers = {"es": Faker("es_ES"), "en": Faker("en_US")}
    for fake in fakers.values():
        fake.seed_instance(seed)
    for index in range(count):
        language = "es" if index % 2 == 0 else "en"
        address = fakers[language].address().replace("\n", ", ")
        prefix = ("Para la entrega confidencial, registra este domicilio: " if language == "es"
                  else "For the confidential delivery, record this postal address: ")
        text = prefix + address + "."
        start = len(prefix)
        yield {"source_text": text, "privacy_mask": [{"start": start, "end": start + len(address),
                                                       "label": "ADDRESS", "value": address}],
               "language": language, "region": "ES" if language == "es" else "US", "source": "synthetic",
               "uid": f"address-challenge-{index}", "template_id": f"address-challenge-v1-{language}"}


def prepare_address_challenge(baseline_dir, output_dir, tokenizer, count=1000, seed=42):
    baseline_dir, output_dir = Path(baseline_dir).resolve(), Path(output_dir).resolve()
    if output_dir == baseline_dir or baseline_dir in output_dir.parents:
        raise ValueError("Output must be separate from the baseline dataset")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    baseline = verify_dataset(baseline_dir)
    categories = [name[2:] for name in baseline["label_names"] if name.startswith("B-")]
    params = {"kind": "independent_address_challenge_v1", "seed": seed, "requested_documents": count,
              "baseline_manifest_sha256": hashlib.sha256((baseline_dir / "manifest.json").read_bytes()).hexdigest()}
    return prepare_records({"challenge": address_challenge_rows(count, seed)}, tokenizer, output_dir,
                           max_length=baseline["max_length"], overlap=baseline["overlap"], seed=seed,
                           categories=categories, source_parameters=params,
                           source_metadata=[{"id": "local-deterministic-synthetic", "license": "Apache-2.0"}])
