"""EGMS-QA translator modeling: projector, batch construction, loss, evaluation.

Shared building blocks for the translator. ``train.py`` imports these to jointly
train the ``EGMSProjector`` and a QLoRA adapter on a frozen host language model:
cross-entropy on answer tokens only, with
inputs_embeds = [projector(tile tokens) ; embed(question)].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class EGMSProjector(nn.Module):
    def __init__(self, egms_dim, llm_hidden, dropout=0.05):
        super().__init__()
        hidden = max(egms_dim, llm_hidden)
        self.net = nn.Sequential(nn.Linear(egms_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, llm_hidden))

    def forward(self, x):
        return self.net(x)


def _response_instruction(r):
    instruction = str(r.get("response_instruction", "")).strip()
    return f"\nResponse format: {instruction}" if instruction else ""


def build_prompt(r):
    return f"Question: {r['question']}{_response_instruction(r)}\nAnswer: {r['answer']}"


def build_prompt_q(r):
    return f"Question: {r['question']}{_response_instruction(r)}\nAnswer:"


def _focus_token_start(text, offsets, focus_text, search_start=0):
    """Return the first token overlapping an exact, renderer-owned span."""
    start = text.find(focus_text, search_start)
    if start < 0:
        raise ValueError(f"loss focus text {focus_text!r} is absent from rendered answer")
    end = start + len(focus_text)
    overlapping = [
        index
        for index, (tok_start, tok_end) in enumerate(offsets)
        if tok_end > start and tok_start < end
    ]
    if not overlapping:
        raise ValueError(f"no tokenizer span overlaps loss focus text {focus_text!r}")
    return overlapping[0]


def _semantic_token_weights(text, offsets, focus_text, weight, search_start=0):
    """Weight tokens overlapping an exact, renderer-owned semantic span."""
    if weight < 1.0:
        raise ValueError(f"semantic token weight must be >= 1, got {weight}")
    start = text.find(focus_text, search_start)
    if start < 0:
        raise ValueError(f"loss focus text {focus_text!r} is absent from rendered answer")
    end = start + len(focus_text)
    weights = [weight if tok_end > start and tok_start < end else 1.0
               for tok_start, tok_end in offsets]
    if weight > 1.0 and not any(value > 1.0 for value in weights):
        raise ValueError(f"no tokenizer span overlaps loss focus text {focus_text!r}")
    return weights


def _select_prompt_end_hidden(last_hidden, label_starts, n_prefix):
    """Select the last question-side hidden state, immediately before the answer."""
    prompt_end = n_prefix + label_starts - 1
    prompt_end = prompt_end.clamp(min=0, max=last_hidden.shape[1] - 1)
    batch_index = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_index, prompt_end]


def _select_focus_start_hidden(last_hidden, focus_starts, n_prefix):
    """Select the causal state that predicts the first token of the focus span."""
    focus_predictor = n_prefix + focus_starts - 1
    focus_predictor = focus_predictor.clamp(min=0, max=last_hidden.shape[1] - 1)
    batch_index = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_index, focus_predictor]


def build_batch(rows, spatial, tok_mask, tid2idx, tokenizer, device, max_prompt,
                semantic_token_weight=1.0, track_focus_positions=False):
    if semantic_token_weight < 1.0:
        raise ValueError(f"semantic token weight must be >= 1, got {semantic_token_weight}")
    n = len(rows)
    n_prefix = spatial.shape[1]
    idxs = [tid2idx[str(r["tile_id"])] for r in rows]
    prefix = spatial[idxs].to(device)
    prefix_mask = tok_mask[idxs].to(device).long()

    full_ids, ans_starts, token_weights, focus_starts = [], [], [], []
    use_semantic_weights = semantic_token_weight > 1.0
    eos = tokenizer.eos_token_id
    for r in rows:
        q_text = build_prompt_q(r)
        full_text = build_prompt(r)
        q_only = tokenizer(q_text, add_special_tokens=False)["input_ids"]
        focus_text = str(r.get("loss_focus_text", "")).strip()
        need_offsets = (use_semantic_weights or track_focus_positions) and bool(focus_text)
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=need_offsets,
        )
        full = encoded["input_ids"]
        weights = [1.0] * len(full)
        focus_start = -1
        if need_offsets:
            if "offset_mapping" not in encoded:
                raise RuntimeError("focus-span supervision requires a fast tokenizer with offset mappings")
            focus_start = _focus_token_start(
                full_text, encoded["offset_mapping"], focus_text, search_start=len(q_text)
            )
            if use_semantic_weights:
                weights = _semantic_token_weights(
                    full_text,
                    encoded["offset_mapping"],
                    focus_text,
                    semantic_token_weight,
                    search_start=len(q_text),
                )
        trim = max(0, len(full) - max_prompt)
        if len(full) > max_prompt:
            full = full[trim:]
            weights = weights[trim:]
        if focus_start >= 0:
            focus_start -= trim
            if focus_start < 0 or focus_start >= len(full):
                raise ValueError(f"loss focus text {focus_text!r} was removed by prompt truncation")
        if eos is not None:
            full = full + [eos]
            weights = weights + [1.0]
        full_ids.append(full)
        token_weights.append(weights)
        ans_starts.append(max(0, min(len(q_only) - trim, len(full))))
        focus_starts.append(focus_start)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    L = max(len(f) for f in full_ids)
    pids = torch.full((n, L), pad_id, dtype=torch.long, device=device)
    pmask = torch.zeros((n, L), dtype=torch.long, device=device)
    loss_weights = torch.ones((n, L), dtype=torch.float32, device=device)
    lstart = torch.zeros(n, dtype=torch.long, device=device)
    slen = torch.zeros(n, dtype=torch.long, device=device)
    fstart = torch.full((n,), -1, dtype=torch.long, device=device)
    for i, (ids, st, weights, focus_start) in enumerate(
        zip(full_ids, ans_starts, token_weights, focus_starts)
    ):
        pids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        pmask[i, :len(ids)] = 1
        loss_weights[i, :len(ids)] = torch.tensor(weights, dtype=torch.float32, device=device)
        lstart[i] = st
        slen[i] = len(ids)
        fstart[i] = focus_start
    batch = {"prefix": prefix, "prefix_mask": prefix_mask, "prompt_ids": pids,
             "prompt_mask": pmask, "label_starts": lstart, "seq_lengths": slen,
             "n_prefix": n_prefix}
    if use_semantic_weights:
        batch["loss_weights"] = loss_weights
    if track_focus_positions:
        batch["focus_starts"] = fstart
    return batch


def forward_loss(batch, projector, qwen, return_prompt_hidden=False, return_focus_hidden=False):
    prefix_proj = projector(batch["prefix"].to(torch.bfloat16))                 # (B,65,H)
    prompt_emb = qwen.get_input_embeddings()(batch["prompt_ids"]).to(torch.bfloat16)
    inputs_embeds = torch.cat([prefix_proj, prompt_emb], dim=1)
    attn = torch.cat([batch["prefix_mask"], batch["prompt_mask"]], dim=1)
    fkw = {}
    if getattr(qwen, "_needs_tti", False):   # Gemma3 multimodal: text token_type_ids (all 0)
        fkw["token_type_ids"] = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    out = qwen(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        use_cache=False,
        output_hidden_states=return_prompt_hidden or return_focus_hidden,
        **fkw,
    )
    logits = out.logits.float()
    n_prefix = batch["n_prefix"]; B, L = batch["prompt_ids"].shape
    labels = torch.full((B, n_prefix + L), -100, dtype=torch.long, device=logits.device)
    for i in range(B):
        a, s = int(batch["label_starts"][i]), int(batch["seq_lengths"][i])
        labels[i, n_prefix + a: n_prefix + s] = batch["prompt_ids"][i, a:s]
    sl = logits[:, :-1, :].contiguous(); slab = labels[:, 1:].contiguous()
    valid = slab != -100
    if "loss_weights" in batch:
        combined_weights = torch.zeros((B, n_prefix + L), dtype=torch.float32, device=logits.device)
        combined_weights[:, n_prefix:] = batch["loss_weights"]
        shifted_weights = combined_weights[:, 1:]
        token_loss = F.cross_entropy(
            sl.reshape(-1, sl.size(-1)),
            slab.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(slab)
        weight_sum = shifted_weights[valid].sum().clamp_min(1.0)
        loss = (token_loss[valid] * shifted_weights[valid]).sum() / weight_sum
    else:
        loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), slab.reshape(-1), ignore_index=-100)
        weight_sum = valid.sum()
    info = {
        "loss": float(loss.item()),
        "valid_tokens": int(valid.sum().item()),
        "loss_weight_sum": float(weight_sum.item()),
    }
    if return_prompt_hidden:
        if not out.hidden_states:
            raise RuntimeError("model did not return hidden states for auxiliary supervision")
        info["prompt_hidden"] = _select_prompt_end_hidden(
            out.hidden_states[-1], batch["label_starts"], n_prefix
        )
    if return_focus_hidden:
        if not out.hidden_states:
            raise RuntimeError("model did not return hidden states for auxiliary supervision")
        if "focus_starts" not in batch:
            raise ValueError("focus positions were not tracked while building the batch")
        info["focus_hidden"] = _select_focus_start_hidden(
            out.hidden_states[-1], batch["focus_starts"], n_prefix
        )
    return loss, info


def evaluate(rows, spatial, tok_mask, tid2idx, tokenizer, projector, qwen, device, args):
    projector.eval(); qwen.eval(); tot_l = 0.0; tot_t = 0
    with torch.no_grad():
        for s in range(0, len(rows), args.batch_size):
            b = build_batch(rows[s:s + args.batch_size], spatial, tok_mask, tid2idx,
                            tokenizer, device, args.max_prompt_tokens)
            _, info = forward_loss(b, projector, qwen)
            tot_l += info["loss"] * info["valid_tokens"]; tot_t += info["valid_tokens"]
    projector.train(); qwen.train()
    avg = tot_l / max(tot_t, 1)
    return {"loss": avg, "ppl": float(math.exp(min(avg, 20.0))), "n_examples": len(rows), "n_tokens": tot_t}


def per_task_eval_loss(rows, spatial, tok_mask, tid2idx, tokenizer, projector, qwen, device, args, cap=400):
    """Answer-token loss broken out per task (cheap signal of which groups are learning)."""
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    projector.eval(); qwen.eval(); out = {}
    with torch.no_grad():
        for task, trows in sorted(by.items()):
            trows = trows[:cap]; tl = 0.0; tt = 0
            for s in range(0, len(trows), args.batch_size):
                b = build_batch(trows[s:s + args.batch_size], spatial, tok_mask, tid2idx,
                                tokenizer, device, args.max_prompt_tokens)
                _, info = forward_loss(b, projector, qwen)
                tl += info["loss"] * info["valid_tokens"]; tt += info["valid_tokens"]
            out[task] = round(tl / max(tt, 1), 4)
    projector.train(); qwen.train()
    return out
