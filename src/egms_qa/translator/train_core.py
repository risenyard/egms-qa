"""EGMS-QA translator training core (short-answer, all A-X tasks).

Jointly trains the EGMSProjector and a QLoRA adapter on a frozen host language
model. The base model is 4-bit and frozen; only the LoRA deltas and the
projector receive gradient. Cross-entropy is applied on answer tokens only, with
inputs_embeds = [projector(tile tokens) ; embed(question)]. LoRA is applied to
all linear layers with gradient checkpointing.

This module holds the reusable building blocks (projector, batch construction,
loss, evaluation); train.py drives sampling and the training loop.

Outputs (per run dir): projector.pt, the LoRA adapter, train/eval logs, summary.
"""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from egms_qa.paths import ENCODER_TOKENS, DEFAULT_HOST_MODEL, QA_DIR as _QA_DIR, OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_DEFAULT = DEFAULT_HOST_MODEL
QA_DIR = str(_QA_DIR)
TOK_CACHE = str(ENCODER_TOKENS)
PHASEA_PROJECTOR = str(OUTPUTS_DIR / "phaseA/projector.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-path", default=QWEN_DEFAULT)
    p.add_argument("--token-cache", default=TOK_CACHE)
    p.add_argument("--qa-train", default=f"{QA_DIR}/curriculum1_qa_train.jsonl")
    p.add_argument("--qa-eval", default=f"{QA_DIR}/curriculum1_qa_eval.jsonl")
    p.add_argument("--include-tasks", default="",
                   help="comma list to restrict tasks; empty = all A-X tasks")
    p.add_argument("--resume-adapter", default="",
                   help="path to a saved qwen_lora_adapter dir to CONTINUE training "
                        "(chains 14h jobs to accumulate epochs); pair with "
                        "--warm-start-projector <same ckpt>/projector.pt")
    p.add_argument("--warm-start-projector", default=PHASEA_PROJECTOR,
                   help="Phase A projector.pt to warm-start; 'none' to train from scratch")
    p.add_argument("--output-dir", default="outputs/runs/default")
    # optimisation
    p.add_argument("--lr", type=float, default=2e-4, help="LoRA lr")
    p.add_argument("--projector-lr", type=float, default=2e-4, help="projector lr (warm-started)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None, help="cap total micro-steps (overrides epochs)")
    p.add_argument("--warmup-steps", type=int, default=200)
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--projector-dropout", type=float, default=0.05)
    # misc
    p.add_argument("--max-prompt-tokens", type=int, default=128)
    p.add_argument("--eval-every-steps", type=int, default=1000)
    p.add_argument("--save-every-steps", type=int, default=4000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=1000)
    p.add_argument("--cls-only", action="store_true",
                   help="ablation: keep only the CLS token as visual prefix "
                        "(drop the 64 patch tokens); eval auto-detects via saved args")
    return p.parse_args()


class EGMSProjector(nn.Module):
    def __init__(self, egms_dim, llm_hidden, dropout=0.05):
        super().__init__()
        hidden = max(egms_dim, llm_hidden)
        self.net = nn.Sequential(nn.Linear(egms_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, llm_hidden))

    def forward(self, x):
        return self.net(x)


def load_qa(path, tasks, split):
    """tasks: set of task ids, or None/empty -> all tasks. Skip c1-into-c2 mix copies."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("mixed_from_curriculum1"):
                continue
            if r["split"] != split:
                continue
            if tasks and r["task"] not in tasks:
                continue
            rows.append(r)
    return rows


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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training (Qwen-9B forward)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    include = {t.strip() for t in args.include_tasks.split(",") if t.strip()}
    print(f"include tasks: {sorted(include) if include else 'ALL (Curriculum-1)'}", flush=True)

    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    print(f"loading Qwen 4bit={not args.no_4bit} from {args.qwen_path}", flush=True)
    qwen = None
    for cls_name in ["Gemma3ForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        if not hasattr(transformers, cls_name):
            continue
        try:
            qwen = getattr(transformers, cls_name).from_pretrained(args.qwen_path, **kw)
            print(f"loaded via {cls_name}", flush=True); break
        except Exception as e:
            print(f"{cls_name} failed: {repr(e)[:160]}", flush=True)
    if qwen is None:
        raise RuntimeError("could not load Qwen")
    llm_hidden = qwen.get_input_embeddings().embedding_dim

    # ---- QLoRA: freeze base, attach all-linear LoRA (proven repo recipe) ----
    # NB: do NOT call prepare_model_for_kbit_training — it upcasts the final norm /
    # lm_head to fp32, which clashes with our bf16 inputs_embeds at the lm_head
    # (RuntimeError: mat1 BFloat16 != mat2 float). The repo's proven QLoRA path
    # (train_qwen35_egms_lora_v1.py) keeps everything bf16 and just enables
    # gradient checkpointing + input-require-grads.
    from peft import LoraConfig, get_peft_model, PeftModel
    if hasattr(qwen, "gradient_checkpointing_enable"):
        qwen.gradient_checkpointing_enable()
    if hasattr(qwen, "enable_input_require_grads"):
        qwen.enable_input_require_grads()
    # multimodal base (e.g. Gemma3ForConditionalGeneration): restrict LoRA to the
    # LANGUAGE-MODEL linears — "all-linear" would also wrap the unused vision tower
    # (wasted params + a bad-module-name peft error). Text-only base -> all-linear.
    has_vision = any("vision" in n for n, _ in qwen.named_modules())
    tgt = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
           if has_vision else "all-linear")
    print(f"LoRA target_modules: {'language-model regex (multimodal base)' if has_vision else 'all-linear'}", flush=True)
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout, bias="none",
                             target_modules=tgt, task_type="CAUSAL_LM")
    if args.resume_adapter and args.resume_adapter.lower() != "none":
        qwen = PeftModel.from_pretrained(qwen, args.resume_adapter, is_trainable=True)
        print(f"RESUMED LoRA adapter from {args.resume_adapter} (continuing training)", flush=True)
    else:
        qwen = get_peft_model(qwen, lora_config)
    qwen._needs_tti = has_vision          # Gemma3 needs token_type_ids during loss
    qwen.config.use_cache = False
    qwen.train()
    qwen.print_trainable_parameters()
    lora_params = [p for p in qwen.parameters() if p.requires_grad]
    print(f"Qwen hidden={llm_hidden}  LoRA tensors={len(lora_params)}  "
          f"GPU={torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)

    # ---- token cache + projector (warm-start from Phase A) ----
    cache = torch.load(args.token_cache, map_location="cpu", weights_only=False)
    spatial = cache["spatial_tokens"].to(torch.float32)
    tok_mask = cache["token_mask"]
    if args.cls_only:
        spatial = spatial[:, :1]
        tok_mask = tok_mask[:, :1]
        print("CLS-ONLY ablation: visual prefix sliced to 1 token (CLS)", flush=True)
    tid2idx = {str(t): i for i, t in enumerate(cache["tile_ids"])}
    egms_dim = spatial.shape[-1]
    print(f"token cache {tuple(spatial.shape)} egms_dim={egms_dim}", flush=True)

    projector = EGMSProjector(egms_dim, llm_hidden, args.projector_dropout).to(device, torch.bfloat16)
    if args.warm_start_projector and args.warm_start_projector.lower() != "none":
        ck = torch.load(args.warm_start_projector, map_location="cpu", weights_only=False)
        assert ck["egms_dim"] == egms_dim and ck["llm_hidden"] == llm_hidden, \
            f"projector dim mismatch: ckpt {ck['egms_dim']}/{ck['llm_hidden']} vs {egms_dim}/{llm_hidden}"
        projector.load_state_dict(ck["projector_state"])
        print(f"warm-started projector from {args.warm_start_projector} "
              f"(Phase A ppl={ck.get('metrics', {}).get('ppl', 'n/a')})", flush=True)
    else:
        print("projector trained from scratch (no warm start)", flush=True)
    projector.train()

    # ---- data ----
    train_rows = load_qa(args.qa_train, include, "train")
    val_rows = load_qa(args.qa_eval, include, "val")
    rng = random.Random(args.seed); rng.shuffle(train_rows)
    if args.max_train_examples:
        train_rows = train_rows[:args.max_train_examples]
    if args.max_val_examples:
        # keep val balanced-ish across tasks by interleaving before cap
        rng.shuffle(val_rows); val_rows = val_rows[:args.max_val_examples]
    tasks_present = sorted({r["task"] for r in train_rows})
    print(f"train={len(train_rows)} val={len(val_rows)} tasks={tasks_present}", flush=True)
    if not train_rows:
        raise RuntimeError("no train rows — check --qa-train / --include-tasks")

    # ---- optimiser: separate groups for projector and LoRA ----
    opt = torch.optim.AdamW([
        {"params": list(projector.parameters()), "lr": args.projector_lr, "weight_decay": args.weight_decay},
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])
    micro_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    n_steps = micro_per_epoch * args.epochs
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    def lr_scale(step):
        if args.warmup_steps and step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        return 1.0
    base_lrs = [g["lr"] for g in opt.param_groups]
    print(f"total micro-steps {n_steps} ({micro_per_epoch}/epoch x {args.epochs})", flush=True)

    tl = open(out / "train_log.jsonl", "w"); el = open(out / "eval_log.jsonl", "w")
    best_ppl = float("inf"); best = None; gstep = 0; t0 = time.monotonic(); stop = False

    def save_ckpt(tag, metrics):
        d = out / tag; d.mkdir(parents=True, exist_ok=True)
        torch.save({"projector_state": projector.state_dict(), "args": vars(args),
                    "metrics": metrics, "egms_dim": egms_dim, "llm_hidden": llm_hidden},
                   d / "projector.pt")
        qwen.save_pretrained(d / "qwen_lora_adapter")

    def run_eval(epoch):
        nonlocal best_ppl, best
        m = evaluate(val_rows, spatial, tok_mask, tid2idx, tokenizer, projector, qwen, device, args)
        m["step"] = gstep; m["epoch"] = epoch
        m["per_task_loss"] = per_task_eval_loss(val_rows, spatial, tok_mask, tid2idx,
                                                tokenizer, projector, qwen, device, args)
        el.write(json.dumps(m) + "\n"); el.flush()
        print(f"  EVAL step {gstep}: loss={m['loss']:.4f} ppl={m['ppl']:.3f}", flush=True)
        print(f"    per-task loss: {m['per_task_loss']}", flush=True)
        if m["ppl"] < best_ppl:
            best_ppl = m["ppl"]; best = m; save_ckpt("best", m)
            print(f"  saved best ppl={best_ppl:.3f}", flush=True)

    opt.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        if stop:
            break
        rng.shuffle(train_rows)
        for s in range(0, len(train_rows), args.batch_size):
            b = build_batch(train_rows[s:s + args.batch_size], spatial, tok_mask, tid2idx,
                            tokenizer, device, args.max_prompt_tokens)
            loss, info = forward_loss(b, projector, qwen)
            (loss / args.grad_accum_steps).backward()
            if (gstep + 1) % args.grad_accum_steps == 0:
                sc = lr_scale(gstep)
                for g, bl in zip(opt.param_groups, base_lrs):
                    g["lr"] = bl * sc
                torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + lora_params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            gstep += 1
            if gstep % args.log_every == 0:
                el_t = time.monotonic() - t0
                print(f"  step {gstep}/{n_steps} ep{epoch} loss={info['loss']:.4f} "
                      f"lr={opt.param_groups[1]['lr']:.2e} {el_t:.0f}s", flush=True)
                tl.write(json.dumps({"step": gstep, "epoch": epoch, "loss": info["loss"],
                                     "elapsed": el_t}) + "\n"); tl.flush()
            if gstep % args.eval_every_steps == 0:
                run_eval(epoch)
            if args.save_every_steps and gstep % args.save_every_steps == 0:
                save_ckpt("last", {"step": gstep})
            if gstep >= n_steps:
                stop = True; break

    fm = evaluate(val_rows, spatial, tok_mask, tid2idx, tokenizer, projector, qwen, device, args)
    fm["step"] = gstep
    fm["per_task_loss"] = per_task_eval_loss(val_rows, spatial, tok_mask, tid2idx,
                                             tokenizer, projector, qwen, device, args)
    el.write(json.dumps({"final": True, **fm}) + "\n"); el.close(); tl.close()
    save_ckpt("last", fm)
    if fm["ppl"] < best_ppl:
        best = fm; best_ppl = fm["ppl"]; save_ckpt("best", fm)
    (out / "summary.json").write_text(json.dumps({
        "best_metrics": best, "final_metrics": fm, "args": vars(args),
        "n_train": len(train_rows), "n_val": len(val_rows),
        "tasks_present": tasks_present, "warm_start": args.warm_start_projector,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": "all-linear"},
        "global_step": gstep, "wall_time_seconds": round(time.monotonic() - t0, 2)}, indent=2))
    print(f"\nwrote {out}/{{best,last}}/{{projector.pt, qwen_lora_adapter}} + summary.json  "
          f"best ppl={best_ppl:.3f}", flush=True)


if __name__ == "__main__":
    main()
