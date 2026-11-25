# Training

Trains two copies of the tetrahedron-3m model from Wang et al. 2024. Every setting matches between them except the training data:

| Run | Data |
|---|---|
| `tetrahedron-3m-aligned-full` | `timaeus/dsir-pile-13m`, all rows |
| `tetrahedron-3m-aligned-nogithub` | The same rows with the GitHub rows removed |

Comparing these two runs isolates the effect of GitHub data on the model. The hyperparameters, and where each one comes from, are in [HYPERPARAMETERS.md](HYPERPARAMETERS.md).

## Setup
```
uv sync
uv run modal setup   # once, to authenticate with Modal
```
The scripts run on Modal and install their own dependencies there. Locally you only need `modal`.

## Data pipeline
`prepare_data.py` turns the 12.8M rows of `timaeus/dsir-pile-13m` into 5,000,000 training sequences of 1024 tokens (50,000 steps × batch 100):

1. **Tokenize.** Each row, a text snippet averaging about 530 tokens, is tokenized with `timaeus/TinyStories-tokenizer-5k` and followed by an EOS token. This step runs once and both variants share it.
2. **Shuffle rows.** All rows are shuffled with seed 42. Both variants use the same permutation.
3. **Filter.** `nogithub` drops every row whose `pile_set_name` contains `Github`: 17,849 rows, holding 13.05M tokens (0.19%). Only 6,840 of those rows are pure GitHub; the rest also contain text from other Pile sets. DM Mathematics is kept. `full` drops nothing.
4. **Pack into sequences.** The rows are joined in shuffled order, cut into consecutive 1023-token pieces, and given a BOS token (the same id as EOS) at the front. The first 5M sequences are used in order: step `s` trains on sequences `100·s` to `100·s + 99`.

This format reproduces the original model's training loss; see HYPERPARAMETERS.md.

Because both variants share the row order, the two runs read the same text in the same order, and the no-GitHub run simply skips the GitHub rows. Their sequences are identical up to the first GitHub row; after that the no-GitHub stream runs slightly ahead, by at most 0.19% of tokens. Both runs also share the same initialization (`seed=1`), so the GitHub data is the only systematic difference between them.

Output: `/vol/data/aligned_<variant>_train.npy` on the Modal Volume `data-refined-llc`.

## Running
```
uv run modal run training/prepare_data.py --variant full
uv run modal run training/prepare_data.py --variant nogithub
uv run modal deploy training/train.py::APP
uv run python training/launch.py --run-name tetrahedron-3m-aligned-full --data aligned_full
uv run python training/launch.py --run-name tetrahedron-3m-aligned-nogithub --data aligned_nogithub
```
Each run takes about 10 hours on an A10 and costs about $11. Runs keep going after the local client exits, and they resume from `resume.pt` if Modal preempts them.

## Outputs
Each run writes to `/runs/<run_name>/` on the volume:
- `checkpoints/checkpoint_XXXXXXX.pth`: step 0, then every 100 steps (501 files). These are TransformerLens state dicts in the same format as `timaeus/tetrahedron-3m-og`.
- `losses.txt`: the training loss at every step.
- `resume.pt`: model and optimizer state, saved every 500 steps.
- `DONE`: written when the run finishes.

To download: `uv run modal volume get data-refined-llc /runs/<run_name> ./runs/`

The volume also holds `tetrahedron-3m-nogithub`, an earlier no-GitHub run stopped at about step 42k. It used a pipeline that shuffled sequences instead of rows, so it is not aligned with the runs above.
