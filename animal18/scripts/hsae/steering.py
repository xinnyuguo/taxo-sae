"""Activation-space steering and LM generation helpers."""
import torch


def add_hook(lm, layer, fn):
    """Hook the block whose OUTPUT is hidden_states[layer] (the tensor the SAE was trained on).
    HF hidden_states[0] is the embedding output, so hidden_states[L] = output of transformer.h[L-1].
    (Before 2026-10-02 this hooked transformer.h[layer], i.e. hidden_states[layer+1]: an off-by-one.)"""
    return lm.transformer.h[layer - 1].register_forward_hook(fn)


def generate_with_hook(lm, tok, inputs, layer, hook_fn, max_new_tokens=20):
    """Greedy decode with a forward hook on `layer`; returns the decoded continuation."""
    hook = add_hook(lm, layer, hook_fn) if hook_fn is not None else None
    try:
        with torch.no_grad():
            o = lm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tok.eos_token_id)
    finally:
        if hook is not None: hook.remove()
    return tok.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def beam_candidates(lm, tok, inputs, layer, hook_fn, k, max_new_tokens=10):
    """Top-k DISTINCT beam candidates (beam width >= 5) — the exact-set@|U| view."""
    hook = add_hook(lm, layer, hook_fn) if hook_fn is not None else None
    try:
        with torch.no_grad():
            o = lm.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=max(k, 5),
                            num_return_sequences=max(k, 5), do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if hook is not None: hook.remove()
    n_in = inputs["input_ids"].shape[1]
    outs = [tok.decode(x[n_in:], skip_special_tokens=True).strip().split("\n")[0] for x in o]
    seen, uniq = set(), []
    for c in outs:
        if c not in seen: seen.add(c); uniq.append(c)
    return uniq[:k]


def last_token_hidden(lm, tok, question, layer, device):
    inputs = tok(f"Q: {question}\nA:", return_tensors="pt").to(device)
    with torch.no_grad():
        h = lm(**inputs, output_hidden_states=True).hidden_states[layer][0, -1, :].unsqueeze(0)
    return inputs, h
