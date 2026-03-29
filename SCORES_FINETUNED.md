# Finetuned Eval: Molmo-7B-D-WaltonSFT

## Model
- **Model**: `scw71432/Molmo-7B-D-WaltonSFT` (1 epoch SFT on `yosubshin/oumi-walton-exclude-geometry`)
- **Dataset**: `WaltonMultimodalColdStart` (10,000 questions)
- **Judge**: `meta-llama/Llama-3.1-8B-Instruct` (via vLLM, alias `llama31-8b`)
- **Eval date**: 2026-03-29

## Score
| Metric | Value |
|--------|-------|
| Correct (verdict=1) | 7,696 / 10,000 |
| **Accuracy** | **76.96%** |
| Incorrect (verdict=0) | 2,304 / 10,000 |

## Notes
- Inference: vLLM, tensor_parallel_size=4, max_output_tokens=8192, batch_size=32
- Judging: vLLM judge with `disable_mm_preprocessor_cache=True`
- Results file: `results/molmo_finetuned_judge.xlsx`
