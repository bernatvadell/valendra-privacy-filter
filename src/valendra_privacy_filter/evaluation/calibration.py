"""Token temperature scaling fitted exclusively on validation labels."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from valendra_privacy_filter.models import PrivacyFilterModel
from valendra_privacy_filter.datasets.prepare import verify_dataset


def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1, logits.shape[-1]).detach()
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    keep = labels != -100
    logits, labels = logits[keep], labels[keep]
    if len(labels) == 0:
        raise ValueError("Validation has no supervised tokens")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 100.0)
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 100.0))


@torch.inference_mode()
def _validation_logits(model, path, device, max_tokens):
    emissions, labels, total = [], [], 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["input_ids", "attention_mask", "labels"]):
        for row in batch.to_pylist():
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
            output = model(input_ids=ids, attention_mask=mask).logits[0].float().cpu()
            valid = torch.tensor(row["labels"]) != -100
            emissions.append(output[valid])
            labels.append(torch.tensor(row["labels"])[valid])
            total += int(valid.sum())
            if total >= max_tokens:
                return torch.cat(emissions)[:max_tokens], torch.cat(labels)[:max_tokens]
    if not labels:
        raise ValueError("Validation dataset is empty")
    return torch.cat(emissions), torch.cat(labels)


def calibrate_checkpoint(checkpoint, dataset_dir, output, device="cpu", max_tokens=100_000):
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    path = Path(dataset_dir) / "validation.parquet"
    verify_dataset(dataset_dir, ["validation"])
    manifest = (Path(dataset_dir) / "manifest.json").read_bytes()
    model = PrivacyFilterModel.from_pretrained(checkpoint, torch_dtype=torch.float32, use_safetensors=True).to(device).eval()
    logits, labels = _validation_logits(model, path, device, max_tokens)
    temperature = fit_temperature(logits, labels)
    result = {"method": "token_temperature_scaling", "temperature": temperature,
              "validation_supervised_tokens": int(len(labels)),
              "dataset_manifest_sha256": hashlib.sha256(manifest).hexdigest()}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
