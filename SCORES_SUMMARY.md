# Molmo-7B SFT Pipeline — Final Results

## Summary Table

| | Baseline | Finetuned |
|---|---|---|
| **Model** | `oumi-ai/Molmo-7B-D-0924` | `scw71432/Molmo-7B-D-WaltonSFT` |
| **Accuracy** | **78.03%** | **76.96%** |
| **Correct** | 7,803 / 10,000 | 7,696 / 10,000 |
| **Delta** | — | -1.07% |

**Dataset**: `WaltonMultimodalColdStart` (10,000 questions)
**Judge**: `meta-llama/Llama-3.1-8B-Instruct` (vLLM, alias `llama31-8b`)

## Pipeline Steps

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment setup (`vlm-eval` conda, torch 2.6+cu124, flash-attn 2.6.3) | ✅ |
| 2 | Baseline eval — `oumi-ai/Molmo-7B-D-0924` | ✅ 78.03% |
| 3 | SFT training — 1 epoch, `yosubshin/oumi-walton-exclude-geometry`, 4×H100 | ✅ loss=0.7290 |
| 4 | Finetuned eval — `scw71432/Molmo-7B-D-WaltonSFT` | ✅ 76.96% |
| 5 | Summary | ✅ (this file) |

## Notes
- Training: TRL_SFT, Adafactor, cosine LR, 1 epoch, ~508s, ~1980 tokens/s, 1000 steps
- Finetuned checkpoint: https://huggingface.co/scw71432/Molmo-7B-D-WaltonSFT
- The -1.07% delta is likely due to training on `oumi-walton-exclude-geometry` (no geometry),
  while the eval set includes geometry questions, causing regression on excluded categories.
