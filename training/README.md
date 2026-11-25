# No-GitHub retrain of tetrahedron-3m

Retrains `timaeus/tetrahedron-3m-og` (the model in Wang et al. 2024) with GitHub data removed, for a like-for-like comparison.

## Steps
1. `modal run training/prepare_data.py` downloads `timaeus/dsir-pile-13m` and drops every row whose `pile_set_name` contains `Github`. DM Mathematics is kept. It then tokenizes the rows with `timaeus/TinyStories-tokenizer-5k`, joins them with EOS, cuts the stream into 1023-token chunks, prepends BOS, shuffles with seed 42 and keeps 5,000,000 sequences.
2. `modal run --detach training/train.py` runs 50,000 steps at batch 100 in fp32 on an A10. It writes `checkpoint_XXXXXXX.pth` every 100 steps (501 files, TransformerLens state dicts in the same format as the original) and `losses.txt` to the Modal Volume `data-refined-llc` under `/runs/tetrahedron-3m-nogithub/`. It resumes from `resume.pt` if it is preempted.

## Fidelity to the original
- The hyperparameters come from `timaeus-research/paper-rllcs-2024` (`figures/*/training.yaml`, `remaining.yaml`).
- Sequence format: on held-out rows, the original final checkpoint scores 3.22 on BOS-prefixed EOS-joined chunks, matching its training loss of about 3.2. Every other format I tried scores 3.37 to 3.53.
- A 1,000-step trial tracks the original `losses.txt`. The 100-step mean loss was 5.67 vs. 5.76 at step 100, 4.50 vs. 4.53 at step 500, and 4.32 vs. 4.33 at step 1,000.
- Known differences: the original was trained on TPUs and I trained on a GPU in strict fp32, the data order differs, and the TransformerLens init with `seed=1` matches the original's statistics but not its exact values.
- The filter removed 17,849 rows, which matches `timaeus/dsir-pile-13m-filtered-for-github`. That is only about 0.14% of rows.
