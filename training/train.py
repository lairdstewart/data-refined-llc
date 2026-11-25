"""Train a tetrahedron-3m model (2-layer attention-only transformer) on Modal.

Settings match timaeus/tetrahedron-3m-og; see HYPERPARAMETERS.md for sources.
Writes checkpoints every 100 steps and losses.txt to /runs/<run_name>/ on the
`data-refined-llc` volume, and resumes from resume.pt after preemption.

Deploy, then start runs with launch.py (see README.md):
    uv run modal deploy training/train.py::APP
Short interactive check (cancelled if the client exits):
    uv run modal run training/train.py --run-name validate --max-steps 1000 --variant aligned_full
"""
import modal

APP = modal.App("drllc-train")
VOL = modal.Volume.from_name("data-refined-llc", create_if_missing=True)
IMG = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "torch==2.5.1", "transformers==4.46.3", "transformer_lens==2.11.0"
)

RUNS = "/vol/runs"
BATCH, MICRO = 100, 25
SAVE_EVERY, RESUME_EVERY = 100, 500


def make_model():
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=256, n_ctx=1024, d_head=32, n_heads=8, d_vocab=5000,
        attn_only=True, act_fn="gelu", positional_embedding_type="shortformer",
        normalization_type="LN", seed=1, device="cuda",
    )
    return HookedTransformer(cfg)


@APP.function(image=IMG, volumes={"/vol": VOL}, gpu="A10", memory=16384,
              timeout=24 * 3600, retries=modal.Retries(max_retries=5, initial_delay=10.0))
def train(run_name: str = "tetrahedron-3m-nogithub", max_steps: int = 50_000, variant: str = "nogithub"):
    import json, os, time
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # strict fp32, as in the original
    run_dir = f"{RUNS}/{run_name}"
    ckpt_dir = f"{run_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = f"{run_dir}/resume.pt"

    data_path = f"/vol/data/{variant}_train.npy"  # see module docstring
    data = np.load(data_path, mmap_mode="r")
    assert data.shape[0] >= max_steps * BATCH and data.shape[1] == 1024, data.shape

    model = make_model()
    # HF Trainer (used for the original) decays every TransformerLens parameter:
    # none are named "bias" and TL's LayerNorm is not torch.nn.LayerNorm.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    step, losses = 0, []
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location="cuda")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        step, losses = st["step"], st["losses"]
        print(f"resumed at step {step}", flush=True)
    else:
        torch.save(model.state_dict(), f"{ckpt_dir}/checkpoint_{0:07d}.pth")
        json.dump({"max_steps": max_steps, "batch": BATCH, "data": data_path, "variant": variant}, open(f"{run_dir}/config.json", "w"))

    model.train()
    t0 = time.time()
    while step < max_steps:
        batch = torch.from_numpy(data[step * BATCH:(step + 1) * BATCH].astype(np.int64)).cuda()
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(0, BATCH, MICRO):
            loss = model(batch[j:j + MICRO], return_type="loss") * (MICRO / BATCH)
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        losses.append(tot)
        if step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"{ckpt_dir}/checkpoint_{step:07d}.pth")
            with open(f"{run_dir}/losses.txt", "w") as f:
                f.write("\n".join(f"{x}" for x in losses) + "\n")
            el = time.time() - t0
            print(f"step {step} loss {np.mean(losses[-100:]):.4f} {el:.0f}s elapsed", flush=True)
        if step % RESUME_EVERY == 0 or step == max_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "losses": losses},
                       resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)
            VOL.commit()
    open(f"{run_dir}/DONE", "w").write(f"{step}\n")
    VOL.commit()
    return {"steps": step, "final_loss_100": float(np.mean(losses[-100:]))}


@APP.local_entrypoint()
def main(run_name: str = "tetrahedron-3m-nogithub", max_steps: int = 50_000, variant: str = "nogithub"):
    print(train.remote(run_name, max_steps, variant))
