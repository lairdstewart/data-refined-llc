"""Start a training run on the deployed `drllc-train` app.

The run executes on Modal and keeps going after this script exits.
Deploy first (and again after changing train.py):
    uv run modal deploy training/train.py::APP

Usage:
    uv run python training/launch.py --run-name tetrahedron-3m-aligned-full --data aligned_full
"""
import argparse

import modal

p = argparse.ArgumentParser()
p.add_argument("--run-name", required=True, help="output folder under /runs/ on the volume")
p.add_argument("--data", required=True, help="data file /vol/data/<data>_train.npy, e.g. aligned_full")
p.add_argument("--max-steps", type=int, default=50_000)
a = p.parse_args()

call = modal.Function.from_name("drllc-train", "train").spawn(a.run_name, a.max_steps, a.data)
print(f"started {a.run_name} on {a.data}: function call {call.object_id}")
