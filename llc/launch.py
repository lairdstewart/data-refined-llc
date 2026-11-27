"""Start an LLC stage on the deployed `drllc-llc` app; it keeps running after this exits.

    uv run modal deploy llc/estimate.py
    uv run python llc/launch.py --dataset pile --heads 6,7
    uv run python llc/launch.py --dataset github --heads all
"""
import argparse

import modal

p = argparse.ArgumentParser()
p.add_argument("--model", default="full", choices=["full", "no-GitHub"])
p.add_argument("--dataset", required=True, choices=["pile", "github"])
p.add_argument("--layer", type=int, default=1)
p.add_argument("--heads", required=True, help="comma-separated head indices, or 'all'")
a = p.parse_args()
heads = list(range(8)) if a.heads == "all" else [int(h) for h in a.heads.split(",")]
call = modal.Function.from_name("drllc-llc", "run_stage").spawn(a.model, a.dataset, a.layer, heads)
print(f"started {a.model}/{a.dataset} L{a.layer} heads {heads}: function call {call.object_id}")
