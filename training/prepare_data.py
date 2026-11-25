"""Build the no-GitHub training set for the tetrahedron-3m retrain.

Source: timaeus/dsir-pile-13m (the data behind timaeus/tetrahedron-3m-og).
Filter: drop every row whose metadata.pile_set_name contains "Github".
Format (matches the original, see README): tokenize each row with
timaeus/TinyStories-tokenizer-5k, join rows with EOS, cut the stream into
1023-token chunks and prepend BOS (= EOS id), giving 1024-token sequences.
Chunks are shuffled with seed 42 and the first 50,000 x 100 are kept, in
training order.

Run:  modal run training/prepare_data.py
"""
import modal

APP = modal.App("drllc-prepare-data")
VOL = modal.Volume.from_name("data-refined-llc", create_if_missing=True)
IMG = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "requests==2.32.3", "transformers==4.46.3", "tokenizers==0.20.3"
)

SRC_URL = "https://huggingface.co/datasets/timaeus/dsir-pile-13m/resolve/main/train_1.jsonl"
RAW = "/vol/raw/train_1.jsonl"
TOK_DIR = "/vol/tok_nogithub"
OUT = "/vol/data/nogithub_train.npy"
TOKENIZER = "timaeus/TinyStories-tokenizer-5k"
DROP_SET = "Github"
N_SHARDS = 64
CTX = 1024
N_STEPS, BATCH = 50_000, 100
SEED = 42


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
    import json, os
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id
    size = os.path.getsize(RAW)
    start, end = size * i // N_SHARDS, size * (i + 1) // N_SHARDS
    texts, kept, dropped = [], 0, 0
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
            if DROP_SET in row["metadata"]["pile_set_name"]:
                dropped += 1
                continue
            kept += 1
            texts.append(row["contents"])
    out = []
    for j in range(0, len(texts), 1000):
        for ids in tok(texts[j : j + 1000])["input_ids"]:
            out.extend(ids)
            out.append(eos)
    arr = np.asarray(out, dtype=np.uint16)
    os.makedirs(TOK_DIR, exist_ok=True)
    np.save(f"{TOK_DIR}/shard_{i:03d}.npy", arr)
    VOL.commit()
    return {"shard": i, "kept": kept, "dropped": dropped, "tokens": int(arr.size)}


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=3600, cpu=4, memory=65536)
def assemble():
    import json, os
    import numpy as np
    from transformers import AutoTokenizer

    eos = AutoTokenizer.from_pretrained(TOKENIZER).eos_token_id

    VOL.reload()
    stream = np.concatenate([np.load(f"{TOK_DIR}/shard_{i:03d}.npy") for i in range(N_SHARDS)])
    L = CTX - 1
    n_chunks = stream.size // L
    need = N_STEPS * BATCH
    assert n_chunks >= need, (n_chunks, need)
    chunks = stream[: n_chunks * L].reshape(n_chunks, L)
    order = np.random.default_rng(SEED).permutation(n_chunks)[:need]
    data = np.empty((need, CTX), dtype=np.uint16)
    data[:, 0] = eos
    data[:, 1:] = chunks[order]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, data)
    meta = {"total_tokens": int(stream.size), "n_chunks": int(n_chunks), "kept_chunks": need, "seed": SEED}
    json.dump(meta, open(OUT.replace(".npy", "_meta.json"), "w"))
    VOL.commit()
    return meta


@APP.local_entrypoint()
def main():
    import json

    print("downloaded bytes:", download.remote())
    stats = list(tokenize_shard.map(range(N_SHARDS)))
    kept = sum(s["kept"] for s in stats)
    dropped = sum(s["dropped"] for s in stats)
    print(f"rows kept {kept:,}, dropped (Github) {dropped:,}, tokens {sum(s['tokens'] for s in stats):,}")
    meta = assemble.remote()
    meta.update(rows_kept=kept, rows_dropped_github=dropped)
    print(json.dumps(meta, indent=1))
