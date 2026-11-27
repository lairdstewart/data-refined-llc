"""Build the fixed SGLD data pools (25,600 sequences of 1024 tokens each).

  pile    timaeus/dsir-pile-1m-2 (rows 10m-11m of the DSIR Pile; the paper's
          `pile1m` eval set)
  github  codeparrot/github-code, all languages (the paper's `github-all`)

Documents are shuffled with seed 0, tokenized with EOS appended, joined, cut
into 1023-token chunks and prefixed with BOS, the same format as training.

Run:  uv run modal run llc/build_pools.py
Output: /vol/llc/pools/<dataset>.npy (uint16) and <dataset>_meta.json
"""
import modal

from common import CTX, IMG, POOL_SEQS, TOKENIZER, VOL, pool_path

APP = modal.App("drllc-pools")
PILE_URL = "https://huggingface.co/datasets/timaeus/dsir-pile-1m-2/resolve/main/train-1m-b.jsonl"
GH_URL = "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-{:05d}-of-01126.parquet"
N_GH_SHARDS, SEED = 8, 0


def _pack(docs, tok, dataset, extra):
    import collections, json, os
    import numpy as np

    eos = tok.eos_token_id
    L, need = CTX - 1, POOL_SEQS
    body, n_docs = [], 0
    total = 0
    for j in range(0, len(docs), 1000):
        for ids in tok([d for d, _ in docs[j:j + 1000]])["input_ids"]:
            body.extend(ids); body.append(eos)
            n_docs += 1
        total = len(body)
        if total >= need * L:
            break
    assert total >= need * L, f"{dataset}: only {total} tokens"
    arr = np.asarray(body[:need * L], dtype=np.uint16).reshape(need, L)
    data = np.concatenate([np.full((need, 1), eos, dtype=np.uint16), arr], axis=1)
    os.makedirs(os.path.dirname(pool_path(dataset)), exist_ok=True)
    np.save(pool_path(dataset), data)
    meta = {"dataset": dataset, "docs_used": n_docs, "sequences": need, "seed": SEED, **extra,
            "label_counts": dict(collections.Counter(lbl for _, lbl in docs[:n_docs]).most_common(40))}
    json.dump(meta, open(pool_path(dataset).replace(".npy", "_meta.json"), "w"), indent=1)
    VOL.commit()
    return meta


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=3600, cpu=8, memory=32768)
def build(dataset: str):
    import io, json, os, random
    import requests
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)
    if dataset == "pile":
        docs = []
        with requests.get(PILE_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if len(docs) >= 200_000:
                    break
                row = json.loads(line)
                docs.append((row["contents"], "|".join(row["metadata"]["pile_set_name"])))
        rng.shuffle(docs)
        return _pack(docs, tok, dataset, {"source": PILE_URL, "rows_read": 200_000})
    import pyarrow.parquet as pq
    shards = sorted(rng.sample(range(1126), N_GH_SHARDS))
    docs = []
    for s in shards:
        b = requests.get(GH_URL.format(s), timeout=600).content
        t = pq.read_table(io.BytesIO(b))
        code_col = next(c for c in ("code", "content", "text") if c in t.column_names)
        print(s, t.column_names, t.num_rows)
        if "language" in t.column_names:
            langs = t.column("language").to_pylist()
        else:  # some shards lack the column; label by file extension
            langs = [os.path.splitext(p)[1].lower() or "none" for p in t.column("path").to_pylist()]
        docs += list(zip(t.column(code_col).to_pylist(), langs))
    rng.shuffle(docs)
    return _pack(docs, tok, dataset, {"source": "codeparrot/github-code", "shards": shards})


@APP.local_entrypoint()
def main(datasets: str = "pile,github"):
    import json
    for m in build.map(datasets.split(","), return_exceptions=True):
        print(m if isinstance(m, Exception) else json.dumps(m, indent=1))
