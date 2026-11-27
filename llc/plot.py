"""Plot per-head refined LLCs in the layout of Wang et al. 2024 Figure 3.

    uv run modal volume get --force data-refined-llc /llc/results results
    uv run python llc/plot.py --results results --out fig3.png [--heads 6,7]

Panels: wrLLC on the Pile, drLLC on GitHub, and (GitHub - Pile) / Pile in %,
each drawn only when its data exists.
"""
import argparse, glob, json, os

import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--results", default="results")
p.add_argument("--model", default="full")
p.add_argument("--layer", type=int, default=1)
p.add_argument("--heads", default="all")
p.add_argument("--induction", default="", help="comma-separated induction-head indices, for color")
p.add_argument("--out", default="fig3.png")
a = p.parse_args()
heads = range(8) if a.heads == "all" else [int(h) for h in a.heads.split(",")]
induction = {int(h) for h in a.induction.split(",") if h}


def series(ds, h):
    files = sorted(glob.glob(os.path.join(a.results, a.model, ds, f"L{a.layer}H{h}", "step_*.json")))
    rows = [json.load(open(f)) for f in files]
    return {r["step"]: r["llc"] for r in rows}


data = {ds: {h: series(ds, h) for h in heads} for ds in ("pile", "github")}
panels = [ds for ds in ("pile", "github") if any(data[ds][h] for h in heads)]
if set(panels) == {"pile", "github"}:
    panels.append("rel")
fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
induction_colors = ["#c0392b", "#e67e22", "#8e44ad"]
other_colors = plt.cm.Blues([0.35 + 0.1 * i for i in range(8)])
titles = {"pile": "The Pile (wrLLC)", "github": "GitHub (drLLC)", "rel": "Relative difference (%)"}
for ax, panel in zip(axes[0], panels):
    for i, h in enumerate(heads):
        ind = h in induction
        color = induction_colors[sorted(induction).index(h) % 3] if ind else other_colors[i]
        if panel == "rel":
            pile, gh = data["pile"][h], data["github"][h]
            xs = sorted(set(pile) & set(gh) - {0})
            ys = [100 * (gh[s] - pile[s]) / pile[s] for s in xs]
        else:
            s = data[panel][h]
            xs = sorted(k for k in s if k > 0)
            ys = [s[k] for k in xs]
        ax.plot(xs, ys, color=color, lw=2.0 if ind else 1.2, label=f"{a.layer}:{h}" + (" (induction)" if ind else ""))
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_title(titles[panel])
    if panel == "rel":
        ax.axhline(0, color="gray", lw=0.8)
    else:
        ax.set_ylabel("refined LLC")
axes[0][0].legend(fontsize=8, frameon=False)
fig.suptitle(f'Layer {a.layer} per-head refined LLCs, "{a.model}" model')
fig.tight_layout()
fig.savefig(a.out, dpi=200)
print("wrote", a.out)
