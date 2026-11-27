# Refined LLCs

Per-head weight-refined LLCs (wrLLC, on the Pile) and weight- and data-refined LLCs (drLLC, on GitHub) for the layer-1 attention heads, as in Wang et al. 2024 Figure 3.

## Setup
Same as `training/`: `uv sync`, then `uv run modal setup` once. `plot.py` runs locally and uses matplotlib.

## Method
The SGLD settings copy `figures/drllcs/llc.yaml` in `timaeus-research/paper-rllcs-2024`:

| Setting | Value |
|---|---|
| Loss | Per-token KL divergence from the checkpoint's own predictions (`kl_div: true`), so the loss at w* is 0 |
| Step size ε | 1e-3 |
| nβ | 30 |
| Localization γ | 200 |
| Chains × draws | 4 × 200, no burn-in |
| SGLD batch | 32 sequences of 1024 tokens |
| Checkpoints | The paper's 107 steps: 0, 100, every 200 to 10k, every 400 to 20k, every 1000 to 50k |

For head h, SGLD moves only that head's slices of `W_Q`, `W_K`, `W_V`, `W_O`, `b_Q`, `b_K` and `b_V`, and the estimate is nβ times the mean KL over all draws. Because the head sits in the last layer, `estimate.py` recomputes only that head and reuses one reference forward pass per batch for every head in a call. `validate` checks this against a full TransformerLens forward pass.

Data are fixed across every estimate: each pool holds 25,600 sequences, and chain c, draw t always uses batch `c·200 + t`, with noise seeded `100 + c`.

| Pool | Source |
|---|---|
| `pile` | `timaeus/dsir-pile-1m-2` (rows 10m to 11m of the DSIR Pile, the paper's `pile1m`) |
| `github` | `codeparrot/github-code`, 8 random shards, all languages (the paper's `github-all`) |

Documents are shuffled with seed 0 and packed like the training data: EOS-joined, 1023-token chunks, with BOS prepended.

## Running
```
uv run modal run llc/build_pools.py
uv run modal run llc/estimate.py::show_induction_scores --model-name full --steps 50000
uv run modal deploy llc/estimate.py
uv run python llc/launch.py --dataset pile --heads 2,7
uv run python llc/launch.py --dataset github --heads all
```
`launch.py` runs one A10 call per checkpoint and skips results that already exist.

## Outputs
On the volume `data-refined-llc`:
- `/llc/pools/<dataset>.npy` and `<dataset>_meta.json`
- `/llc/results/<model>/<dataset>/L1H<h>/step_XXXXXX.json`: the estimate, per-chain values and loss traces
- `/llc/summaries/*.json`: GPU seconds per stage

To plot:
```
uv run modal volume get --force data-refined-llc /llc/results results
uv run python llc/plot.py --results results --induction 2 --out fig3.png
```
