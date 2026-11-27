"""Shared Modal objects and constants for refined-LLC estimation.

SGLD settings copy figures/drllcs/llc.yaml in timaeus-research/paper-rllcs-2024
(see llc/README.md). Data pools and chain seeds are fixed, so every estimate
(any checkpoint, head or stage) uses the same batches and the same noise seeds.
"""
import modal

VOL = modal.Volume.from_name("data-refined-llc", create_if_missing=True)
IMG = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "torch==2.5.1", "transformers==4.46.3", "transformer_lens==2.11.0",
    "pyarrow==18.1.0", "requests==2.32.3",
).add_local_python_source("common")

TOKENIZER = "timaeus/TinyStories-tokenizer-5k"
CTX = 1024

# SGLD, from llc.yaml
EPSILON = 1e-3          # lr
NBETA = 30.0            # nbeta_sgld = nbeta_estimate
GAMMA = 200.0           # localization
NUM_CHAINS = 4
NUM_DRAWS = 200         # num_burnin_steps 0, num_steps_bw_draws 1
BATCH = 32
INIT_SEED = 100
NOISE_LEVEL = 1.0

POOL_SEQS = NUM_CHAINS * NUM_DRAWS * BATCH  # 25,600: each batch used once per estimate

# Paper's eval-step grid (figures/drllcs/llc.yaml `steps`), 107 checkpoints.
STEPS = [0, 100] + list(range(200, 10_000, 200)) + list(range(10_000, 20_000, 400)) + list(range(20_000, 50_001, 1000))
assert len(STEPS) == 107

RUNS = {"full": "/vol/runs/tetrahedron-3m-aligned-full", "no-GitHub": "/vol/runs/tetrahedron-3m-aligned-nogithub",
        "full-oginit": "/vol/runs/tetrahedron-3m-oginit-full",
        "full-oginit2": "/vol/runs/tetrahedron-3m-oginit2-full"}
DATASETS = ("pile", "github")


def pool_path(dataset: str) -> str:
    return f"/vol/llc/pools/{dataset}.npy"


def result_path(model: str, dataset: str, layer: int, head: int, step: int) -> str:
    return f"/vol/llc/results/{model}/{dataset}/L{layer}H{head}/step_{step:06d}.json"


def make_model(device: str = "cuda"):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2, d_model=256, n_ctx=1024, d_head=32, n_heads=8, d_vocab=5000,
        attn_only=True, act_fn="gelu", positional_embedding_type="shortformer",
        normalization_type="LN", seed=1, device=device,
    )
    return HookedTransformer(cfg)
