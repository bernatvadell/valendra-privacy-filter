"""Globally constrained BIOES Viterbi decoding."""

import torch


def _parts(label):
    if label == "O":
        return "O", None
    prefix, category = label.split("-", 1)
    if prefix not in "BIES" or not category:
        raise ValueError(f"Invalid BIOES label: {label}")
    return prefix, category


def viterbi_decode(logits, names):
    if logits.ndim != 2 or logits.shape[1] != len(names):
        raise ValueError("Expected logits with shape (tokens, labels)")
    if logits.shape[0] == 0:
        return []
    parts = [_parts(name) for name in names]
    scores = logits.detach().float().cpu()
    n = len(names)
    transitions = torch.full((n, n), -torch.inf)
    for i, (previous, old_type) in enumerate(parts):
        for j, (current, new_type) in enumerate(parts):
            if ((previous in {"O", "E", "S"} and current in {"O", "B", "S"})
                    or (previous in {"B", "I"} and current in {"I", "E"} and old_type == new_type)):
                transitions[i, j] = 0.0
    state = scores[0].clone()
    for i, (prefix, _) in enumerate(parts):
        if prefix not in {"O", "B", "S"}:
            state[i] = -torch.inf
    backpointers = []
    for emission in scores[1:]:
        best, previous = (state[:, None] + transitions).max(dim=0)
        state = best + emission
        backpointers.append(previous)
    for i, (prefix, _) in enumerate(parts):
        if prefix not in {"O", "E", "S"}:
            state[i] = -torch.inf
    if not torch.isfinite(state).any():
        raise ValueError("No valid BIOES path")
    current = int(state.argmax())
    result = [current]
    for previous in reversed(backpointers):
        current = int(previous[current])
        result.append(current)
    return list(reversed(result))
