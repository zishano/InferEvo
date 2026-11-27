"""Small CPU references for semantics the simulator's cost model assumes.

Independent of RoofLang/Simthesizer code (numpy only, float64), following
the structure of Kimi-K3 `modeling_kimi_linear.py` at reduced dimensions.

mla_prefill_vs_absorbed: MLA attention computed by materializing per-head
  K/V from the compressed cache (prefill pricing, CM-5) equals the absorbed
  latent form (decode pricing: q·W_UK in latent space, then W_UV).
kda_chunked_equals_single_pass: KDA (short causal conv + gated delta rule
  with per-channel decay) processed in chunks / token by token, carrying only
  the fixed-size recurrent state and the last (conv_width-1) conv inputs,
  equals one pass over the whole sequence. This justifies P→D handoff of a
  fixed-size KDA state (MC-2) and chunked-prefill state continuity.
"""

from __future__ import annotations

import numpy as np


def mla_prefill_vs_absorbed(seed=0, heads=4, nope=8, rope=4, v_dim=8, kv_lora=6,
                            tokens=7, hidden=16):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, hidden))
    w_dkv = rng.standard_normal((hidden, kv_lora))                 # compress
    w_kr = rng.standard_normal((hidden, rope))                     # shared rope key
    w_uk = rng.standard_normal((heads, kv_lora, nope))             # up-proj K
    w_uv = rng.standard_normal((heads, kv_lora, v_dim))            # up-proj V
    q_nope = rng.standard_normal((heads, tokens, nope))
    q_rope = rng.standard_normal((heads, tokens, rope))
    c = x @ w_dkv                                                   # cache latent
    k_rope = x @ w_kr                                               # cache rope part
    scale = 1.0 / np.sqrt(nope + rope)
    mask = np.triu(np.full((tokens, tokens), -np.inf), 1)

    def softmax(s):
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(-1, keepdims=True)

    # Prefill: materialize per-head K/V (qk = nope+rope, v = v_dim per pair).
    k = np.einsum("tc,hcd->htd", c, w_uk)
    v = np.einsum("tc,hcd->htd", c, w_uv)
    s = (np.einsum("htd,hsd->hts", q_nope, k)
         + np.einsum("htd,sd->hts", q_rope, k_rope)) * scale
    out_prefill = np.einsum("hts,hsd->htd", softmax(s + mask), v)

    # Decode/absorbed: attention in latent space (kv_lora + rope per pair).
    q_lat = np.einsum("htd,hcd->htc", q_nope, w_uk)
    s2 = (np.einsum("htc,sc->hts", q_lat, c)
          + np.einsum("htd,sd->hts", q_rope, k_rope)) * scale
    ctx_lat = np.einsum("hts,sc->htc", softmax(s2 + mask), c)
    out_absorbed = np.einsum("htc,hcd->htd", ctx_lat, w_uv)
    return float(np.max(np.abs(out_prefill - out_absorbed)))


def _kda_step(state, q, k, v, g, beta):
    """Gated delta rule with per-channel decay: S <- diag(exp g) S;
    S <- S + beta k (v - S^T k)^T; o = S^T q. Shapes: S [dk, dv]."""
    state = np.exp(g)[:, None] * state
    k = k / np.linalg.norm(k)
    q = q / np.linalg.norm(q)
    state = state + beta * np.outer(k, v - state.T @ k)
    return state, state.T @ q


def kda_chunked_equals_single_pass(seed=0, tokens=13, hidden=12, dk=6, dv=5,
                                   width=4, chunks=(5, 4, 1, 1, 1, 1),
                                   carry_conv=True, carry_state=True):
    """Chunked prefill then single-token decode vs one pass; max abs diff.

    carry_conv / carry_state=False are negative controls: dropping either part
    of the KDA state at chunk boundaries must change the output.
    """
    assert sum(chunks) == tokens
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, hidden))
    proj = (rng.standard_normal((hidden, dk)), rng.standard_normal((hidden, dk)),
            rng.standard_normal((hidden, dv)), rng.standard_normal((hidden, dk)),
            rng.standard_normal((hidden, 1)))
    conv = rng.standard_normal((3, max(dk, dv), width))

    def fresh():
        return (np.zeros((dk, dv)),
                {"q": np.zeros((width - 1, dk)), "k": np.zeros((width - 1, dk)),
                 "v": np.zeros((width - 1, dv))})

    full, _, _ = _kda_run_streams(x, proj, conv, *fresh())
    state, tail = fresh()
    pieces, start = [], 0
    for n in chunks:
        if not carry_conv:
            tail = fresh()[1]
        if not carry_state:
            state = fresh()[0]
        out, state, tail = _kda_run_streams(x[start:start + n], proj, conv, state, tail)
        pieces.append(out)
        start += n
    return float(np.max(np.abs(full - np.concatenate(pieces))))


def _kda_run_streams(x, proj_params, conv, state, conv_tail):
    """Same as _kda_run but with per-stream conv widths (q/k: dk, v: dv)."""
    wq, wk, wv, wg, wb = proj_params
    width = conv.shape[-1]
    proj = {"q": x @ wq, "k": x @ wk, "v": x @ wv}
    conv_out, tails = {}, {}
    for i, n in enumerate(("q", "k", "v")):
        seq = np.concatenate([conv_tail[n], proj[n]])
        w = conv[i, :seq.shape[1]]                       # [channels, width]
        y = np.stack([(seq[t:t + width] * w.T).sum(0)
                      for t in range(len(seq) - width + 1)])
        conv_out[n] = y / (1 + np.exp(-y))
        tails[n] = seq[-(width - 1):]
    g = -np.logaddexp(0, -(x @ wg)) - 0.1
    beta = 1 / (1 + np.exp(-(x @ wb)))
    outputs = []
    for t in range(len(x)):
        state, o = _kda_step(state, conv_out["q"][t], conv_out["k"][t],
                             conv_out["v"][t], g[t], beta[t, 0])
        outputs.append(o)
    return np.stack(outputs), state, tails
