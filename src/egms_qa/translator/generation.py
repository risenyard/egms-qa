"""EGMS-QA decode utilities (task-agnostic, A-X task families).

Shared by the evaluator: build_prompt_q, parse_number, match_region,
greedy_decode_one. The projector itself lives in ``modeling.py``.
"""
from __future__ import annotations
import re
import torch

REGIONS = ["northwest", "northeast", "southwest", "southeast",
           "north", "south", "east", "west", "center", "centre"]


def build_prompt_q(r):
    instruction = str(r.get("response_instruction", "")).strip()
    suffix = f"\nResponse format: {instruction}" if instruction else ""
    return f"Question: {r['question']}{suffix}\nAnswer:"


def parse_number(text):
    """Prefer the value after 'about' (skips ordinals like '10th'/'90th'); else last number."""
    t = text.replace(",", "")
    m = re.search(r"about\s+(-?\d+\.?\d*)", t)
    if m:
        return float(m.group(1))
    nums = re.findall(r"-?\d+\.?\d*", t)
    return float(nums[-1]) if nums else None


def match_region(text):
    t = text.lower()
    for r in REGIONS:  # longest-first ordering in the list (northwest before north)
        if re.search(r"\b" + r + r"\b", t):
            return "center" if r == "centre" else r
    return None


def greedy_decode_one(row, spatial, tok_mask, tid2idx, tokenizer, projector, host_model, device, max_new):
    """Single-example greedy decode (no padding → no position_id ambiguity). Proven path."""
    idx = tid2idx[str(row["tile_id"])]
    prefix = spatial[idx:idx + 1].to(device)
    prefix_mask = tok_mask[idx:idx + 1].to(device).long()
    prompt_ids = tokenizer(build_prompt_q(row), add_special_tokens=False,
                           return_tensors="pt")["input_ids"].to(device)
    prefix_proj = projector(prefix.to(torch.bfloat16))
    prompt_emb = host_model.get_input_embeddings()(prompt_ids).to(torch.bfloat16)
    inputs_embeds = torch.cat([prefix_proj, prompt_emb], dim=1)
    attn = torch.cat([prefix_mask, torch.ones_like(prompt_ids)], dim=1)

    eos = tokenizer.eos_token_id
    gen = []
    out = host_model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1)
    gen.append(nxt.item())
    for _ in range(max_new - 1):
        if gen[-1] == eos:
            gen.pop(); break
        emb = host_model.get_input_embeddings()(nxt.unsqueeze(0)).to(torch.bfloat16)
        attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        out = host_model(inputs_embeds=emb, attention_mask=attn, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1)
        gen.append(nxt.item())
    return tokenizer.decode(gen, skip_special_tokens=True)


if __name__ == "__main__":
    import sys
    sys.exit(
        "eval_training_generation.py is now a PURE decode-utils module (EGMSProjector, "
        "greedy_decode_one, build_prompt_q, parse_number, match_region).\n"
        "The old G-task evaluator __main__ was removed in the A-X refactor.\n"
        "==> For evaluation use egms_qa.translator.evaluate "
        "(python -m egms_qa.translator.evaluate --adapter-dir <dir>)."
    )
