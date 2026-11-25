# No-GitHub retrain of tetrahedron-3m

Retrains `timaeus/tetrahedron-3m-og` (the model in Wang et al. 2024) with GitHub data removed, for a like-for-like comparison.

## Steps
1. `modal run training/prepare_data.py --variant nogithub` and `--variant full` build two row-aligned datasets from `timaeus/dsir-pile-13m`. Every row is tokenized with `timaeus/TinyStories-tokenizer-5k` and followed by EOS. All 12.8M rows are shuffled with seed 42, using the same permutation for both variants. `nogithub` then drops rows whose `pile_set_name` contains `Github`; DM Mathematics is kept. The rows are joined, cut into consecutive 1023-token sequences with BOS prepended, and the first 5,000,000 sequences are kept in order. Output: `/vol/data/aligned_<variant>_train.npy`.
2. `modal deploy training/train.py::APP`, then `modal.Function.from_name('drllc-train', 'train').spawn(run_name, 50000, 'aligned_<variant>')`, runs 50,000 steps at batch 100 in fp32 on an A10. It writes `checkpoint_XXXXXXX.pth` every 100 steps (501 files, TransformerLens state dicts in the same format as the original) and `losses.txt` to the Modal Volume `data-refined-llc` under `/runs/<run_name>/`. It resumes from `resume.pt` if it is preempted.

## Why the two runs are aligned
Both runs use the same init (`seed=1`) and read the same rows in the same order. The no-GitHub run skips the GitHub rows, so by the end its stream is ahead by at most 0.19% of tokens. The only systematic difference between them is the GitHub data. Wang et al. Appendix G found that seed-to-seed variation can change how many induction heads develop. Shuffling sequences independently in each run would add noise on that scale, so both runs share one ordering instead.

The first run, `tetrahedron-3m-nogithub`, used an earlier pipeline that shuffled sequences rather than rows (commit 2a4e34c). It is not aligned with the control.

## Fidelity to the original
- The hyperparameters come from `timaeus-research/paper-rllcs-2024` (`figures/*/training.yaml`, `remaining.yaml`).
- Sequence format: on held-out rows, the original final checkpoint scores 3.22 on BOS-prefixed EOS-joined chunks, matching its training loss of about 3.2. Every other format I tried scores 3.37 to 3.53.
- A 1,000-step trial tracks the original `losses.txt`. The 100-step mean loss was 5.67 vs. 5.76 at step 100, 4.50 vs. 4.53 at step 500, and 4.32 vs. 4.33 at step 1,000.
- Known differences: the original was trained on TPUs and I trained on a GPU in strict fp32, the data order differs, and the TransformerLens init with `seed=1` matches the original's statistics but not its exact values.
- The filter removed 17,849 rows (0.14% of rows), which matches `timaeus/dsir-pile-13m-filtered-for-github`. Those rows hold 13.05M tokens, or 0.19% of the 6.81B tokens. Only 6,840 of them are pure GitHub; the rest also contain text from other Pile sets.
- See `HYPERPARAMETERS.md` for the source of each setting.
