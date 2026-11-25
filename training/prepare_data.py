"""Build row-aligned training sets for the tetrahedron-3m retrains.

Source: timaeus/dsir-pile-13m (the data behind timaeus/tetrahedron-3m-og).
Variants:
  nogithub  drop every row whose metadata.pile_set_name contains "Github"
  full      keep every row (control run)

Pipeline (both variants share one tokenization pass):
  1. Tokenize every row with timaeus/TinyStories-tokenizer-5k, appending EOS.
  2. Shuffle the 12.8M rows with seed 42. The permutation is over all rows, so
     it is identical for both variants.
  3. Drop the variant's excluded rows, keeping the shuffled order.
  4. Join the rows into one stream, cut it into consecutive 1023-token
     sequences, prepend BOS (= EOS id) and keep the first 50,000 x 100,
     in order. Step s trains on sequences [100 s, 100 s + 100).
Both runs therefore read the same text in the same order; the no-GitHub run
skips the GitHub rows, so its stream runs ahead of the full one by at most
0.19% of tokens.

Run:  uv run modal run training/prepare_data.py --variant nogithub
      uv run modal run training/prepare_data.py --variant full
Output: /vol/data/aligned_<variant>_train.npy  (uint16, [5_000_000, 1024])
"""
import modal

APP = modal.App("drllc-prepare-data")
VOL = modal.Volume.from_name("data-refined-llc", create_if_missing=True)
IMG = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "requests==2.32.3", "transformers==4.46.3", "tokenizers==0.20.3"
)

SRC_URL = "https://huggingface.co/datasets/timaeus/dsir-pile-13m/resolve/main/train_1.jsonl"
RAW = "/vol/raw/train_1.jsonl"
TOK_DIR = "/vol/tok_rows"  # variant-independent: per-row tokens, lengths, Pile sets
TOKENIZER = "timaeus/TinyStories-tokenizer-5k"
VARIANTS = {"nogithub": {"Github"}, "full": set()}  # variant -> Pile sets to drop
N_SHARDS = 64
CTX = 1024
N_STEPS, BATCH = 50_000, 100
SEED = 42


def out_path(variant: str) -> str:
    return f"/vol/data/aligned_{variant}_train.npy"


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=3600, cpu=4)
def download():
    import os, requests
    if os.path.exists(RAW) and os.path.getsize(RAW) > 21_000_000_000:
        return os.path.getsize(RAW)
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    with requests.get(SRC_URL, stream=True, timeout=120) as r, open(RAW + ".part", "wb") as f:
        r.raise_for_status()
        for block in r.iter_content(1 << 24):
            f.write(block)
    os.replace(RAW + ".part", RAW)
    VOL.commit()
    return os.path.getsize(RAW)


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=3600, cpu=2, memory=8192)
def tokenize_shard(i: int):
    """Tokenize every row in byte range i; save tokens, row lengths and Pile sets."""
    import json, os
    import numpy as np
    from transformers import AutoTokenizer

    done = f"{TOK_DIR}/shard_{i:03d}.json"
    if os.path.exists(done):
        return json.load(open(done))
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id
    size = os.path.getsize(RAW)
    start, end = size * i // N_SHARDS, size * (i + 1) // N_SHARDS
    texts, sets = [], []
    with open(RAW, "rb") as f:
        if start:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()  # skip partial line; previous shard owns it
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            row = json.loads(line)
            texts.append(row["contents"])
            sets.append("|".join(row["metadata"]["pile_set_name"]))
    toks, lens = [], []
    for j in range(0, len(texts), 1000):
        for ids in tok(texts[j : j + 1000])["input_ids"]:
            toks.extend(ids)
            toks.append(eos)
            lens.append(len(ids) + 1)
    os.makedirs(TOK_DIR, exist_ok=True)
    np.save(f"{TOK_DIR}/shard_{i:03d}_tokens.npy", np.asarray(toks, dtype=np.uint16))
    np.save(f"{TOK_DIR}/shard_{i:03d}_lens.npy", np.asarray(lens, dtype=np.int32))
    json.dump(sets, open(f"{TOK_DIR}/shard_{i:03d}_sets.json", "w"))
    stats = {"shard": i, "rows": len(lens), "tokens": len(toks)}
    json.dump(stats, open(done, "w"))
    VOL.commit()
    return stats


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=3600, cpu=4, memory=65536)
def assemble(variant: str):
    import json, os
    import numpy as np
    from transformers import AutoTokenizer

    VOL.reload()
    eos = AutoTokenizer.from_pretrained(TOKENIZER).eos_token_id
    drop = VARIANTS[variant]
    stream = np.concatenate([np.load(f"{TOK_DIR}/shard_{i:03d}_tokens.npy") for i in range(N_SHARDS)])
    lens = np.concatenate([np.load(f"{TOK_DIR}/shard_{i:03d}_lens.npy") for i in range(N_SHARDS)]).astype(np.int64)
    sets = [s for i in range(N_SHARDS) for s in json.load(open(f"{TOK_DIR}/shard_{i:03d}_sets.json"))]
    assert len(sets) == lens.size and lens.sum() == stream.size
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    keep = np.array([not (drop & set(s.split("|"))) for s in sets])

    order = np.random.default_rng(SEED).permutation(lens.size)  # same for every variant
    order = order[keep[order]]

    L, need = CTX - 1, N_STEPS * BATCH
    body = np.empty(need * L, dtype=np.uint16)
    pos = 0
    for r in order:
        n = min(lens[r], body.size - pos)
        body[pos : pos + n] = stream[starts[r] : starts[r] + n]
        pos += n
        if pos == body.size:
            break
    assert pos == body.size, f"not enough tokens: {pos} < {body.size}"
    data = np.empty((need, CTX), dtype=np.uint16)
    data[:, 0] = eos
    data[:, 1:] = body.reshape(need, L)
    out = out_path(variant)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, data)
    meta = {"variant": variant, "rows_total": int(lens.size), "rows_kept": int(keep.sum()),
            "rows_dropped": int((~keep).sum()), "tokens_kept": int(lens[keep].sum()),
            "tokens_dropped": int(lens[~keep].sum()), "sequences": need, "seed": SEED}
    json.dump(meta, open(out.replace(".npy", "_meta.json"), "w"))
    VOL.commit()
    return meta


@APP.local_entrypoint()
def main(variant: str = "nogithub"):
    import json

    assert variant in VARIANTS, f"variant must be one of {list(VARIANTS)}"
    print("downloaded bytes:", download.remote())
    stats = list(tokenize_shard.map(range(N_SHARDS)))
    print(f"rows {sum(s['rows'] for s in stats):,}, tokens {sum(s['tokens'] for s in stats):,}")
    print(json.dumps(assemble.remote(variant), indent=1))
