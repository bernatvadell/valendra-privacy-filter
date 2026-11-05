import torch
import pytest
from transformers import Qwen3Config

from valendra_privacy_filter.models.qwen3 import PrivacyFilterModel


def tiny_model(mode="causal", backend="eager"):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        num_labels=41,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = backend
    config.privacy_attention_mode = mode
    return PrivacyFilterModel(config)


def test_logits_loss_backward_and_checkpoint_roundtrip(tmp_path):
    model = tiny_model()
    model.train()
    input_ids = torch.tensor([[2, 3, 4, 0], [6, 7, 0, 0]])
    mask = input_ids.ne(0).long()
    labels = torch.tensor([[0, 1, 2, -100], [0, 3, -100, -100]])
    output = model(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert output.logits.shape == (2, 4, 41)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.model.embed_tokens.weight.grad is not None
    assert model.score.weight.grad is not None
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.eval()
    expected = model(input_ids=input_ids, attention_mask=mask).logits.detach()
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    loaded = PrivacyFilterModel.from_pretrained(tmp_path).eval()
    actual = loaded(input_ids=input_ids, attention_mask=mask).logits.detach()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_causal_and_bidirectional_future_token_influence(backend):
    first = torch.tensor([[2, 3, 4, 5]])
    changed = torch.tensor([[2, 3, 4, 9]])
    mask = torch.ones_like(first)
    causal = tiny_model("causal", backend).eval()
    bidirectional = tiny_model("bidirectional", backend).eval()
    with torch.no_grad():
        c1 = causal(input_ids=first, attention_mask=mask).logits[:, 0]
        c2 = causal(input_ids=changed, attention_mask=mask).logits[:, 0]
        b1 = bidirectional(input_ids=first, attention_mask=mask).logits[:, 0]
        b2 = bidirectional(input_ids=changed, attention_mask=mask).logits[:, 0]
    torch.testing.assert_close(c1, c2, rtol=0, atol=1e-7)
    assert not torch.allclose(b1, b2, rtol=1e-5, atol=1e-7)
    assert bidirectional.config.use_cache is False


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_bidirectional_does_not_attend_to_padding(backend):
    model = tiny_model("bidirectional", backend).eval()
    mask = torch.tensor([[1, 1, 0, 0]])
    with torch.no_grad():
        first = model(input_ids=torch.tensor([[2, 3, 4, 5]]), attention_mask=mask).logits[:, :2]
        second = model(input_ids=torch.tensor([[2, 3, 7, 9]]), attention_mask=mask).logits[:, :2]
    torch.testing.assert_close(first, second, rtol=0, atol=1e-7)
