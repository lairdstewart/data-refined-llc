"""Weight- (and data-) refined LLCs of single attention heads, via SGLD.

For head h in layer l, SGLD moves only that head's slices of W_Q, W_K, W_V,
W_O, b_Q, b_K, b_V (b_O is shared across heads and stays fixed). The loss is
the per-token KL divergence from the checkpoint's own predictions
(`kl_div: true` in the paper's config), so the loss at w* is 0 and
    llc = nbeta * mean_{chains, draws} KL(p_{w*} || p_w).
Each SGLD step:
    w <- w - (eps/2) * (nbeta * grad + gamma * (w - w*)) + sqrt(eps) * N(0, 1)
restricted to the head's weights.

Deploy, then start a stage with launch.py (see llc/README.md):
    uv run modal deploy llc/estimate.py
"""
import modal

from common import (BATCH, CTX, EPSILON, GAMMA, IMG, INIT_SEED, NBETA, NOISE_LEVEL, NUM_CHAINS,
                    NUM_DRAWS, RUNS, STEPS, VOL, make_model, pool_path, result_path)

APP = modal.App("drllc-llc")
HEAD_PARAMS = ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V")


def load(model_name: str, step: int):
    import torch
    m = make_model()
    m.load_state_dict(torch.load(f"{RUNS[model_name]}/checkpoints/checkpoint_{step:07d}.pth", map_location="cuda"))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def reference(m, x, layer: int):
    """One forward pass at w*: what a single-head perturbation needs."""
    import torch
    import torch.nn.functional as F
    names = {f"blocks.{layer}.ln1.hook_normalized", "hook_pos_embed", f"blocks.{layer}.attn.hook_z",
             f"blocks.{layer}.hook_resid_post"}
    with torch.no_grad():
        logits, cache = m.run_with_cache(x, names_filter=lambda n: n in names)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        neg_ent = (p * logp).sum(-1)
    assert layer == m.cfg.n_layers - 1, "fast path assumes the head's layer is the last one"
    return {"n1": cache[f"blocks.{layer}.ln1.hook_normalized"], "pos": cache["hook_pos_embed"],
            "z": cache[f"blocks.{layer}.attn.hook_z"], "resid": cache[f"blocks.{layer}.hook_resid_post"],
            "p": p, "neg_ent": neg_ent}


def head_kl(m, ref, layer: int, h: int, w: dict):
    """KL(p_{w*} || p_w) per token when only head h of `layer` (the last layer) takes weights w.

    The other heads, b_O and every other layer are unchanged, so the final residual
    is resid* - z*_h W_O*_h + z_h W_O_h. Matches the TransformerLens forward pass
    (see `validate`), with q/k input ln1(resid) + pos and v input ln1(resid).
    """
    import torch.nn.functional as F
    attn = m.blocks[layer].attn
    qk_in = ref["n1"] + ref["pos"]
    q = qk_in @ w["W_Q"] + w["b_Q"]
    k = qk_in @ w["W_K"] + w["b_K"]
    v = ref["n1"] @ w["W_V"] + w["b_V"]
    z = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], is_causal=True,
                                       scale=1.0 / attn.attn_scale)[:, 0]
    resid = ref["resid"] - ref["z"][:, :, h] @ attn.W_O[h] + z @ w["W_O"]
    logq = F.log_softmax(m.unembed(m.ln_final(resid)), dim=-1)
    return ref["neg_ent"] - (ref["p"] * logq).sum(-1)


@APP.function(image=IMG, volumes={"/vol": VOL}, gpu="A10", memory=16384, timeout=3600,
              retries=modal.Retries(max_retries=3, initial_delay=5.0))
def estimate(model_name: str, dataset: str, layer: int, heads: list[int], step: int) -> dict:
    """SGLD for every head in `heads` at one checkpoint. Each (head, chain) has its own
    noise generator seeded INIT_SEED + chain and sees the same batches, so results do
    not depend on which heads share a call."""
    import json, math, os, time
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    todo = [h for h in heads if not os.path.exists(result_path(model_name, dataset, layer, h, step))]
    if not todo:
        return {"step": step, "skipped": heads, "seconds": 0.0}
    pool = torch.from_numpy(np.load(pool_path(dataset)).astype(np.int64))
    m = load(model_name, step)
    attn = m.blocks[layer].attn
    w_star = {h: {n: getattr(attn, n)[h].detach().clone() for n in HEAD_PARAMS} for h in todo}
    state = {(h, c): {n: t.clone() for n, t in w_star[h].items()} for h in todo for c in range(NUM_CHAINS)}
    gens = {(h, c): torch.Generator(device="cuda").manual_seed(INIT_SEED + c) for h in todo for c in range(NUM_CHAINS)}
    traces = {(h, c): [] for h in todo for c in range(NUM_CHAINS)}

    for t in range(NUM_DRAWS):
        for c in range(NUM_CHAINS):
            i = (c * NUM_DRAWS + t) * BATCH
            ref = reference(m, pool[i:i + BATCH].cuda(non_blocking=True), layer)
            for h in todo:
                w = state[(h, c)]
                for x in w.values():
                    x.requires_grad_(True)
                loss = head_kl(m, ref, layer, h, w).mean()
                grads = torch.autograd.grad(loss, list(w.values()))
                traces[(h, c)].append(loss.item())
                with torch.no_grad():
                    for (n, x), g in zip(w.items(), grads):
                        x.requires_grad_(False)
                        noise = torch.randn(x.shape, device="cuda", generator=gens[(h, c)])
                        x.add_(-0.5 * EPSILON * (NBETA * g + GAMMA * (x - w_star[h][n]))
                               + math.sqrt(EPSILON) * NOISE_LEVEL * noise)
            del ref

    out = {}
    for h in todo:
        tr = np.asarray([traces[(h, c)] for c in range(NUM_CHAINS)])
        res = {"model": model_name, "dataset": dataset, "layer": layer, "head": h, "step": step,
               "llc": float(NBETA * tr.mean()), "llc_per_chain": (NBETA * tr.mean(1)).tolist(),
               "loss_traces": tr.tolist(), "init_loss": 0.0,
               "config": {"epsilon": EPSILON, "nbeta": NBETA, "gamma": GAMMA, "chains": NUM_CHAINS,
                          "draws": NUM_DRAWS, "batch": BATCH, "ctx": CTX, "init_seed": INIT_SEED, "loss": "kl"}}
        path = result_path(model_name, dataset, layer, h, step)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(res, open(path, "w"))
        out[h] = res["llc"]
    VOL.commit()
    return {"step": step, "llc": out, "seconds": time.time() - t0}


@APP.function(image=IMG, volumes={"/vol": VOL}, gpu="A10", memory=16384, timeout=600)
def validate(model_name: str = "full", step: int = 20_000, layer: int = 1, h: int = 2) -> dict:
    """Check head_kl against a full TransformerLens forward pass with perturbed weights."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.from_numpy(np.load(pool_path("pile"))[:8].astype(np.int64)).cuda()
    m = load(model_name, step)
    ref = reference(m, x, layer)
    torch.manual_seed(0)
    attn = m.blocks[layer].attn
    w = {n: getattr(attn, n)[h] + 0.03 * torch.randn_like(getattr(attn, n)[h]) for n in HEAD_PARAMS}
    fast = head_kl(m, ref, layer, h, w)
    m2 = load(model_name, step)
    with torch.no_grad():
        for n in HEAD_PARAMS:
            getattr(m2.blocks[layer].attn, n)[h] = w[n]
        logq = F.log_softmax(m2(x), dim=-1)
    slow = (ref["p"] * (torch.log(ref["p"].clamp_min(1e-30)) - logq)).sum(-1)
    slow = ref["neg_ent"] - (ref["p"] * logq).sum(-1)
    return {"fast_mean": fast.mean().item(), "slow_mean": slow.mean().item(),
            "max_abs_diff": (fast - slow).abs().max().item(), "zero_check": head_kl(m, ref, layer, h, {n: getattr(attn, n)[h] for n in HEAD_PARAMS}).abs().max().item()}


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=6 * 3600)
def run_stage(model_name: str, dataset: str, layer: int, heads: list[int], steps: list[int] | None = None) -> dict:
    """Fan out one estimate call per checkpoint; write a summary with GPU seconds."""
    import json, os, time
    steps = steps or STEPS
    t0 = time.time()
    rows = list(estimate.map([model_name] * len(steps), [dataset] * len(steps), [layer] * len(steps),
                             [heads] * len(steps), steps, return_exceptions=True))
    errors = [str(r) for r in rows if isinstance(r, Exception)]
    ok = [r for r in rows if not isinstance(r, Exception)]
    summary = {"model": model_name, "dataset": dataset, "layer": layer, "heads": heads,
               "n_steps": len(steps), "n_ok": len(ok), "errors": errors[:5],
               "gpu_seconds_in_function": sum(r["seconds"] for r in ok), "wall_seconds": time.time() - t0}
    os.makedirs("/vol/llc/summaries", exist_ok=True)
    tag = f"{model_name}_{dataset}_L{layer}_H{'-'.join(map(str, heads))}_{int(time.time())}"
    json.dump(summary, open(f"/vol/llc/summaries/{tag}.json", "w"), indent=1)
    VOL.commit()
    return summary


@APP.function(image=IMG, volumes={"/vol": VOL}, cpu=8, memory=16384, timeout=900)
def induction_scores(model_name: str = "full", step: int = 50_000, seq_len: int = 200, n: int = 64) -> dict:
    """Per layer-1 head: prefix-matching score and the loss increase from mean-ablating it.

    Inputs are BOS + a 200-token Pile snippet repeated twice (snippets from the end
    of the Pile pool). The score is the mean attention from each token in the second
    copy to the token after its first occurrence; the ablation cost is the change
    in mean loss on the second copy.
    """
    import os
    import numpy as np
    import torch
    VOL.reload()
    torch.set_num_threads(8)
    path = f"{RUNS[model_name]}/checkpoints/checkpoint_{step:07d}.pth"
    if not os.path.exists(path):
        return {"step": step, "missing": True}
    m = make_model("cpu")
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    pool = torch.from_numpy(np.load(pool_path("pile"), mmap_mode="r")[-n:].astype(np.int64))
    T = seq_len
    x = torch.cat([pool[:, :1], pool[:, 1:T + 1], pool[:, 1:T + 1]], 1)
    q = torch.arange(T + 1, 2 * T + 1)

    def second_copy_loss(hooks=()):
        with torch.no_grad():
            lp = m.run_with_hooks(x, fwd_hooks=list(hooks)).log_softmax(-1)
        return (-lp[:, :-1].gather(-1, x[:, 1:, None])[..., 0])[:, T + 1:].mean().item()

    with torch.no_grad():
        _, cache = m.run_with_cache(x, names_filter=lambda s: s == "blocks.1.attn.hook_pattern")
    score = cache["blocks.1.attn.hook_pattern"][:, :, q, q - T + 1].mean((0, 2)).tolist()
    base = second_copy_loss()
    abl = []
    for h in range(8):
        def f(z, hook, h=h):
            z[:, :, h] = z[:, :, h].mean((0, 1), keepdim=True)
            return z
        abl.append(second_copy_loss([("blocks.1.attn.hook_z", f)]) - base)
    return {"step": step, "score": [round(v, 3) for v in score], "ablation": [round(v, 3) for v in abl],
            "repeat_loss": round(base, 3)}


@APP.local_entrypoint()
def show_induction_scores(model_name: str = "full", steps: str = "50000"):
    import json
    for r in induction_scores.map([model_name] * len(steps.split(",")), [int(s) for s in steps.split(",")]):
        print(json.dumps(r))


@APP.local_entrypoint()
def check():
    print(validate.remote())


@APP.local_entrypoint()
def one(step: int = 20_000, heads: str = "2", dataset: str = "pile", model_name: str = "full"):
    """Single-checkpoint check, e.g. to time an estimate."""
    print(estimate.remote(model_name, dataset, 1, [int(h) for h in heads.split(",")], step))
