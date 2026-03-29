# Baseline Eval: Molmo-7B-D-0924

## Model
- **Model**: `oumi-ai/Molmo-7B-D-0924` (untuned)
- **Dataset**: `WaltonMultimodalColdStart` (10,000 questions)
- **Judge**: `meta-llama/Llama-3.1-8B-Instruct` (via vLLM, alias `llama31-8b`)
- **Eval date**: 2026-03-28

## Score
| Metric | Value |
|--------|-------|
| Correct (verdict=1) | 7,803 / 10,000 |
| **Accuracy** | **78.03%** |
| Incorrect (verdict=0) | 2,197 / 10,000 |

## Notes
- Inference: vLLM, tensor_parallel_size=4, max_output_tokens=8192, batch_size=32
- Judging: vLLM judge with `disable_mm_preprocessor_cache=True`
- Finetuned checkpoint eval still in progress (ETA ~01:20 UTC 2026-03-29)
- Results file: `results/molmo_baseline_judge.xlsx`
