# DCVLR Eval Summary — 2026-03-31

## Overview

Training and evaluating 3 Qwen3-VL-8B-Instruct fine-tuned checkpoints on 10 benchmarks.
All training uses seed 2025, 1 epoch, adafactor optimizer, lr=2e-5, no PEFT, no FSDP (single GPU).

| # | Run Name | HF Dataset | HF Checkpoint | Training | Eval |
|---|---|---|---|---|---|
| 1 | qwen3vl_diverse_cluster | `yosubshin/WaltonMultimodalColdStart-diverse-1k-42` | `scw71432/qwen3vl-8b-diverse-cluster-seed2025` | ✅ | ✅ |
| 4 | qwen3vl_exclude_geo_bio_stats | `yosubshin/oumi-walton-exclude-geometry-biology-statistics` | `scw71432/qwen3vl-8b-exclude-geo-bio-stats-seed2025` | ✅ | ✅ |
| 9 | qwen3vl_hard_exclude_geo_bio_stats | `yosubshin/walton-hard-exclude-geometry-biology-statistics-1k-1` | `scw71432/qwen3vl-8b-hard-exclude-geo-bio-stats-seed2025` | ✅ | 🔄 |

---

## Results Summary

> ⚠️ Physics dataset scores (atomic/electro/mechanics/optics/quantum/statistics) use **exact-match fallback** for all runs — the qwen3-4b judge OOMs alongside vllm. True scores will be higher after re-scoring with the standalone judge.

### Non-Physics Benchmarks

| Benchmark | Run 1 (diverse_cluster) | Run 2 (excl_geo_bio_stats) | Run 3 (hard_excl_geo_bio_stats) |
|---|---|---|---|
| VMCBench_DEV | **77.0%** | **74.9%** | **68.1%** |
| LiveXivTQA | **77.27%** † | **75.87%** | **73.42%** |
| OlympiadBench AVG | **15.05%** | **14.49%** | 🔄 In progress |
| Omni3DBench YN | 52.0% | 49.3% | ⏳ |
| Omni3DBench MC | 62.8% | 62.8% | ⏳ |
| Omni3DBench count | 17.1% | 17.1% | ⏳ |
| Omni3DBench other | 29.1% | 30.9% | ⏳ |

† Run 1 LiveXivTQA scored via regex extraction of `\boxed{X}` (judge OOM'd); Runs 2 & 3 scored by qwen3-4b judge.

### Physics Datasets (⚠️ Exact-match fallback — needs re-scoring)

| Benchmark | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| atomic | 2.5% | 3.0% | ⏳ |
| electro | 1.24% | 0.0% | ⏳ |
| mechanics | 0.0% | 0.0% | ⏳ |
| optics | 1.27% | 0.63% | ⏳ |
| quantum | 1.27% | 1.27% | ⏳ |
| statistics | 1.67% | 1.67% | ⏳ |

---

## Detailed Scores

### VMCBench_DEV Breakdown

| Category | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| **Overall** | **77.0%** | **74.9%** | **68.1%** |
| General | 81.4% | 83.1% | 77.7% |
| Reasoning | 58.7% | 54.7% | 50.3% |
| OCR | 94.0% | 93.0% | 93.0% |
| Doc & Chart | 86.0% | 80.4% | 66.0% |
| SEEDBench | 74.0% | 82.0% | 76.0% |
| MMStar | 66.0% | 78.0% | 64.0% |
| A-OKVQA | 88.0% | 84.0% | 86.0% |
| VizWiz | 84.0% | 82.0% | 86.0% |
| MMVet | 74.0% | 74.0% | 66.0% |
| VQAv2 | 94.0% | 94.0% | 84.0% |
| OKVQA | 90.0% | 88.0% | 82.0% |
| MMMU | 52.0% | 46.0% | 48.0% |
| MathVista | 52.0% | 52.0% | 38.0% |
| ScienceQA | 86.0% | 80.0% | 62.0% |
| RealWorldQA | 46.0% | 46.0% | 50.0% |
| GQA | 76.0% | 72.0% | 76.0% |
| MathVision | 40.0% | 32.0% | 28.0% |
| TextVQA | 90.0% | 92.0% | 90.0% |
| OCRVQA | 98.0% | 94.0% | 96.0% |
| AI2D | 86.0% | 62.0% | 42.0% |
| ChartQA | 88.0% | 86.0% | 68.0% |
| DocVQA | 100.0% | 100.0% | 88.0% |
| InfoVQA | 84.0% | 78.0% | 76.0% |
| TableVQABench | 72.0% | 76.0% | 56.0% |

### OlympiadBench Breakdown

| Subset | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| OE_MM_maths_en_COMP | 23.3% | 25.3% | ⏳ |
| OE_MM_maths_zh_CEE | 14.2% | 11.4% | ⏳ |
| OE_MM_maths_zh_COMP | 12.5% | 12.5% | ⏳ |
| OE_MM_physics_en_COMP | 3.5% | 3.9% | ⏳ |
| OE_MM_physics_zh_CEE | 3.0% | 3.4% | ⏳ |
| OE_TO_maths_en_COMP | 34.8% | 36.4% | ⏳ |
| OE_TO_maths_zh_CEE | 26.3% | 25.8% | ⏳ |
| OE_TO_maths_zh_COMP | 16.7% | 17.2% | ⏳ |
| OE_TO_physics_en_COMP | 5.1% | 4.7% | ⏳ |
| OE_TO_physics_zh_CEE | 2.6% | 0.9% | ⏳ |
| zh_maths | 18.6% | 17.0% | ⏳ |
| zh_physics | 3.0% | 3.3% | ⏳ |
| en_maths | 32.7% | 34.4% | ⏳ |
| en_physics | 4.0% | 4.2% | ⏳ |
| maths | 21.2% | 20.3% | ⏳ |
| physics | 3.3% | 3.5% | ⏳ |
| **AVG** | **15.05%** | **14.49%** | ⏳ |

### Omni3DBench Breakdown

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Yes/No Accuracy | 52.0% | 49.3% | ⏳ |
| Multiple Choice Accuracy | 62.8% | 62.8% | ⏳ |
| Numeric (count) Accuracy | 17.1% | 17.1% | ⏳ |
| Numeric (other) Mean Rel. Accuracy | 29.1% | 30.9% | ⏳ |

---

## Run 3 Status — `qwen3vl_hard_exclude_geo_bio_stats` (🔄 In Progress)

Eval launched 2026-03-30 22:53 UTC. OlympiadBench inference in progress as of 05:31 UTC 2026-03-31.
ETA: ~15:00 UTC 2026-03-31.

---

## Known Issues & Fixes

| Issue | Fix | File |
|---|---|---|
| Custom Qwen3-VL checkpoints not recognized as vllm-compatible → flash_attention OOM | Added `Qwen3VLChat` to `vllm_compatible_classes` | `vlmeval/utils/model_detection.py` |
| Non-MoE checkpoints incorrectly flagged as MoE → vllm init crash | `is_moe_model()` reads `config.json` for `num_experts` first | `vlmeval/vlm/qwen3_vl/model.py` |
| LiveXivTQA scoring OOM when qwen3-4b judge loads alongside vllm | try/except fallback to exact matching | `vlmeval/dataset/image_mcq.py` |
| OlympiadBench scoring crashes on LaTeX parsing | Installed `antlr4-python3-runtime==4.11` | — |

### Physics Scoring Issue
All physics scores are near-zero because the qwen3-4b judge OOMs (~67 GiB vllm footprint) and falls back to strict exact string matching on free-form physics answers. Re-scoring with a standalone judge (no vllm running) is required for meaningful results.

---

## Infrastructure

- **Machine:** 4× NVIDIA H100 80GB HBM3
- **Training env:** `miniconda3/envs/oumi` (oumi 0.7, torch 2.8.0+cu128)
- **Eval env:** `miniconda3/envs/qwen3-eval`
- **Eval framework:** VLMEvalKit (fork: `scwatson4/VLMEvalKit`)
- **Judge:** `qwen3-4b` → `Qwen/Qwen3-4B-Instruct-2507`
- **Inference:** vllm with tensor_parallel_size=4, gpu_memory_utilization=0.7
