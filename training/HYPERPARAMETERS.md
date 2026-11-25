# Hyperparameters and where they come from

Source key:
- **Paper**: Wang et al. 2024, Appendix F.1.
- **Config**: `timaeus-research/paper-rllcs-2024` on GitHub, `figures/*/training.yaml` (a Hugging Face `TrainingArguments` dump with `run_name: tetrahedron-3m-og`) and `figures/*/remaining.yaml` (the model config).
- **Checkpoint**: the published `timaeus/tetrahedron-3m-og` checkpoints on Hugging Face.
- **Inferred**: not stated anywhere. I deduced it from evidence, listed in the table.
- **Choice**: my decision where the original is unknown or can't be matched exactly.

## Model
| Setting | Value | Source |
|---|---|---|
| Architecture | 2-layer attention-only transformer (TransformerLens) | Paper, Config (`attn_only: true`, `implementation: transformer_lens`) |
| d_model / heads / d_head | 256 / 8 / 32 | Paper, Config |
| Context length | 1024 | Paper, Config |
| Vocab / tokenizer | 5000, `TinyStories-tokenizer-5k` | Paper, Config (`georgeyw/...`). I used the `timaeus/...` copy, which gives identical token ids on test strings. |
| Positional embedding | Shortformer, learned | Paper, Config |
| Normalization | LayerNorm with weight and bias (`LN`) | Paper ("included"). Checkpoint keys `ln1.w`/`ln1.b` show it is `LN`, not `LNPre`. |
| act_fn | gelu (unused in an attention-only model) | Config |
| Init | TransformerLens default, `seed=1` | Config (`model_seed: 1`). The statistics match checkpoint 0 (std 0.05), but the exact values don't. |

## Optimization
| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW (torch) | Config (`optim: ADAMW_TORCH`) |
| Learning rate | 1e-3, constant, no warmup | Config (`lr_scheduler_type: CONSTANT`, `warmup_steps: 0`) |
| Betas / eps | 0.9, 0.999 / 1e-8 | Config |
| Weight decay | 0.05, on every parameter | Config for the value. "Every parameter" is inferred: HF Trainer exempts only parameters named "bias" or inside `torch.nn.LayerNorm`, and TransformerLens has neither (`b_Q`, its own LayerNorm class). |
| Grad clipping | max norm 1.0 | Config |
| Batch size | 100 sequences per step | Paper. Config has 25 per device with no accumulation on TPU (`_n_gpu: 0`), which implies 4 devices. I run 4 micro-batches of 25 and average them, which is mathematically identical. |
| Steps | 50,000 | Paper, Config |
| Precision | fp32, TF32 matmuls off | Config (`bf16: false`, `fp16: false`, `tf32: null`) and fp32 checkpoints. Choice: the original ran on TPU, whose default matmul precision differs, so this can't match exactly. |
| Checkpoints | Step 0, then every 100 steps (501 files), TransformerLens state dict | Config (`save_steps: 100`) and Checkpoint (file list and format) |

## Data
| Setting | Value | Source |
|---|---|---|
| Base dataset | `timaeus/dsir-pile-13m` | Inferred. The paper says "a subset of the DSIR-filtered Pile". This is Timaeus's 13M-row DSIR Pile set, and their own no-GitHub run used a filtered copy of it. |
| GitHub filter (`nogithub` variant) | Drop rows whose `pile_set_name` contains `Github` | Choice, per the experiment design. It drops 17,849 rows, the same count as `timaeus/dsir-pile-13m-filtered-for-github`. |
| Sequence format | Rows joined with EOS, cut into 1023-token chunks, BOS (= EOS id) prepended | Inferred. The original final checkpoint scores 3.22 on this format, matching its training loss of about 3.2. Other formats score 3.37 to 3.53. This is also TransformerLens's `tokenize_and_concatenate` convention. |
| Data order | Rows shuffled with seed 42 (one permutation shared by all variants), then joined and cut into sequences, first 5M used in order | Choice. The original order is unknown; `seed: 42` reuses the Config Trainer seed. See README "Data pipeline". |
| Loss | Mean next-token cross-entropy over the 1023 predicted positions | Inferred: TransformerLens `return_type="loss"`. |
