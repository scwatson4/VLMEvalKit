# Molmo-7B SFT Pipeline Progress

## Step 1: Environment Setup ✅
- Conda env: `vlm-eval` (Python 3.10) at `/media/volume/ICML_TESTING_NEW_MODELS/miniconda3/envs/vlm-eval`
- Key packages: torch 2.6.0+cu124, flash-attn 2.6.3, flashinfer, VLMEvalKit deps, judge deps
- Framework: YosubShin/VLMEvalKit fork

## Step 2: Baseline Eval ✅
- Model: `oumi-ai/Molmo-7B-D-0924` (untuned)
- Dataset: `WaltonMultimodalColdStart` (10,000 questions)
- Inference: vLLM, tensor_parallel_size=4, max_output_tokens=8192, batch_size=32
- Judge: `meta-llama/Llama-3.1-8B-Instruct` (alias `llama31-8b`) via vLLM
- **Score: 78.03%** (7,803 / 10,000 correct)
- Results: `results/molmo_baseline_judge.xlsx`

## Step 3: SFT Training ✅
- Config: `configs/dcvlr/molmo_sft.yaml`
- Dataset: `yosubshin/oumi-walton-exclude-geometry`
- Trainer: TRL_SFT, Adafactor, cosine LR, 1 epoch, 4×H100
- Results: 1000 steps, train_loss=0.7290, ~508s, ~1980 tokens/s
- Checkpoint: `/media/volume/ICML_TESTING_NEW_MODELS/checkpoints/molmo_7b_seed2025`

## Step 4: Finetuned Eval 🔄 In Progress
- Model: finetuned checkpoint above
- Inference: 8,380 / 10,000 predictions complete (ETA ~01:20 UTC 2026-03-29)
- Judging will run automatically once inference finishes (fix: `disable_mm_preprocessor_cache=True`)

## Step 5: Summary Table ⏳ Pending
- Will compare baseline vs finetuned scores once judging completes

## Key Fix (this repo)
Added `llama31-8b` → `meta-llama/Llama-3.1-8B-Instruct` shortname mapping in
`_build_vllm_judge()` (`vlmeval/dataset/walton_multimodal.py`) to prevent path
corruption when the judge model name contains `/` characters.
