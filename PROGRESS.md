# Molmo-7B SFT Pipeline Progress

## Step 3: SFT Training ✅
- Config: `configs/dcvlr/molmo_sft.yaml`
- Dataset: `yosubshin/oumi-walton-exclude-geometry`
- Trainer: TRL_SFT, Adafactor, cosine LR, 1 epoch, 4×H100
- Results: 1000 steps, train_loss=0.7290, ~508s, ~1980 tokens/s
- Checkpoint: `/media/volume/ICML_TESTING_NEW_MODELS/checkpoints/molmo_7b_seed2025`

## Step 4: Finetuned Eval 🔄 In Progress
- Model: finetuned checkpoint above
- 760 / 10,000 predictions complete (~10h remaining as of 2026-03-27 23:53 UTC)
- Judging will run automatically via `run_evals_v3.sh` once inference finishes

## Step 5: Summary Table ⏳ Pending
- Will compare baseline vs finetuned scores once judging completes

## Key Fix (this repo)
Added `llama31-8b` → `meta-llama/Llama-3.1-8B-Instruct` shortname mapping in
`_build_vllm_judge()` (`vlmeval/dataset/walton_multimodal.py`) to prevent path
corruption when the judge model name contains `/` characters.
