"""Write the original run's initial weights to the volume.

TransformerLens 1.11 with `seed=1` reproduces timaeus/tetrahedron-3m-og's step-0
weights (that checkpoint is saved after one optimizer step, so they agree to
about one Adam step, 1e-3). TransformerLens 2.x draws W_K, W_V and W_O
differently, so its seed-1 init differs from the original.

Run:  uv run modal run training/make_init.py --seed 1
Output: /vol/init/tl111_seed<seed>.pth, loadable by train.py --init
"""
import modal

APP = modal.App("drllc-make-init")
VOL = modal.Volume.from_name("data-refined-llc", create_if_missing=True)
IMG = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cpu")
       .pip_install("numpy<2", "transformer_lens==1.11.0", "transformers==4.40.2", "pyarrow<15",
                    "pytest", "typeguard<3", "requests"))
OG0 = "https://huggingface.co/timaeus/tetrahedron-3m-og/resolve/main/checkpoints/checkpoint_0000000.pth"


@APP.function(image=IMG, volumes={"/vol": VOL}, cpu=2, memory=8192, timeout=900)
def make(seed: int) -> dict:
    import io, os
    import requests, torch
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=256, n_ctx=1024, d_head=32, n_heads=8, d_vocab=5000,
        attn_only=True, act_fn="gelu", positional_embedding_type="shortformer",
        normalization_type="LN", seed=seed, device="cpu",
    )
    sd = HookedTransformer(cfg).state_dict()
    os.makedirs("/vol/init", exist_ok=True)
    torch.save(sd, f"/vol/init/tl111_seed{seed}.pth")
    VOL.commit()
    og = torch.load(io.BytesIO(requests.get(OG0, timeout=120).content), map_location="cpu")
    return {k: round((og[k] - sd[k]).abs().max().item(), 4) for k in og if sd[k].dtype.is_floating_point and "IGNORE" not in k}


@APP.local_entrypoint()
def main(seed: int = 1):
    d = make.remote(seed)
    print(f"max |og step-0 - init| over tensors: {max(d.values())}  (one Adam step is ~0.001)")
