# DCVLR Eval Summary — 2026-03-30

## Overview

Training and evaluating 3 Qwen3-VL-8B-Instruct fine-tuned checkpoints on 10 benchmarks.
All training uses seed 2025, 1 epoch, adafactor optimizer, lr=2e-5, no PEFT, no FSDP (single GPU).

| # | Run Name | HF Dataset | HF Checkpoint | Training Status |
|---|---|---|---|---|
| 1 | qwen3vl_diverse_cluster | `yosubshin/WaltonMultimodalColdStart-diverse-1k-42` | `scw71432/qwen3vl-8b-diverse-cluster-seed2025` | ✅ Done |
| 4 | qwen3vl_exclude_geo_bio_stats | `yosubshin/oumi-walton-exclude-geometry-biology-statistics` | `scw71432/qwen3vl-8b-exclude-geo-bio-stats-seed2025` | ✅ Done |
| 9 | qwen3vl_hard_exclude_geo_bio_stats | `yosubshin/walton-hard-exclude-geometry-biology-statistics-1k-1` | `scw71432/qwen3vl-8b-hard-exclude-geo-bio-stats-seed2025` | ✅ Done |

---

## Eval Status

### Run 1 — `qwen3vl_diverse_cluster` (✅ Complete)

| Benchmark | Score | Notes |
|---|---|---|
| VMCBench_DEV | **77.0%** | Rule-based scorer |
| LiveXivTQA | **77.27%** | Scored via regex extraction of `\boxed{X}` from predictions (judge OOM'd alongside vllm) |
| OlympiadBench | **15.05%** AVG | math 21.22%, physics 3.32% — rule-based MathJudger |
| Omni3DBench | See below | Rule-based scoring |
| atomic_dataset | **2.5%** ⚠️ | Exact match fallback — needs re-scoring with standalone qwen3-4b judge |
| electro_dataset | **1.24%** ⚠️ | Same |
| mechanics_dataset | **0.0%** ⚠️ | Same |
| optics_dataset | **1.27%** ⚠️ | Same |
| quantum_dataset | **1.27%** ⚠️ | Same |
| statistics_dataset | **1.67%** ⚠️ | Same |

**Omni3DBench breakdown:**
| Metric | Score |
|---|---|
| Yes/No Accuracy | 52.0% |
| Multiple Choice Accuracy | 62.8% |
| Numeric (count) Accuracy | 17.1% |
| Numeric (other) Mean Relative Accuracy | 29.1% |

**VMCBench_DEV breakdown:**
| Category | Score |
|---|---|
| Overall | 77.0% |
| General | 81.4% |
| Reasoning | 58.7% |
| OCR | 94.0% |
| Doc & Chart | 86.0% |
| SEEDBench | 74.0% |
| MMStar | 66.0% |
| A-OKVQA | 88.0% |
| VizWiz | 84.0% |
| MMVet | 74.0% |
| VQAv2 | 94.0% |
| OKVQA | 90.0% |
| MMMU | 52.0% |
| MathVista | 52.0% |
| ScienceQA | 86.0% |
| RealWorldQA | 46.0% |
| GQA | 76.0% |
| MathVision | 40.0% |
| TextVQA | 90.0% |
| OCRVQA | 98.0% |
| AI2D | 86.0% |
| ChartQA | 88.0% |
| DocVQA | 100.0% |
| InfoVQA | 84.0% |
| TableVQABench | 72.0% |

---

### Run 2 — `qwen3vl_exclude_geo_bio_stats` (🔄 In Progress — OlympiadBench inference ~12:03 UTC)

| Benchmark | Score | Notes |
|---|---|---|
| VMCBench_DEV | **74.9%** | ✅ |
| LiveXivTQA | **75.87%** | ✅ qwen3-4b judge |
| OlympiadBench | 🔄 In progress | — |
| Omni3DBench | ⏳ Queued | — |
| atomic_dataset | ⏳ Queued | — |
| electro_dataset | ⏳ Queued | — |
| mechanics_dataset | ⏳ Queued | — |
| optics_dataset | ⏳ Queued | — |
| quantum_dataset | ⏳ Queued | — |
| statistics_dataset | ⏳ Queued | — |

---

### Run 3 — `qwen3vl_hard_exclude_geo_bio_stats` (⏳ Not started)

Checkpoint uploaded to HF. Blocked until Run 2 frees the GPUs.

---

## Known Issues & Fixes Applied

| Issue | Fix | File |
|---|---|---|
| Custom Qwen3-VL checkpoints not recognized as vllm-compatible → flash_attention OOM | Added `Qwen3VLChat` to `vllm_compatible_classes` | `vlmeval/utils/model_detection.py` |
| Non-MoE checkpoints (path has no `4B`/`8B`) incorrectly flagged as MoE → vllm init crash | `is_moe_model()` now reads `config.json` for `num_experts` first | `vlmeval/vlm/qwen3_vl/model.py` |
| LiveXivTQA scoring crashes when qwen3-4b judge can't load (vllm holds all GPU memory) | Added try/except fallback to exact matching in `'qwen' in model` branch | `vlmeval/dataset/image_mcq.py` |
| OlympiadBench scoring crashes on LaTeX parsing | Installed `antlr4-python3-runtime==4.11` in qwen3-eval env | — |

### Physics Dataset Scoring Concern
The physics scores for Run 1 (atomic 2.5%, electro 1.24%, mechanics 0.0%, optics 1.27%, quantum 1.27%, statistics 1.67%) are almost certainly **not** reflective of true model performance.
The qwen3-4b judge OOMs when trying to load alongside the vllm inference model (~67 GiB GPU usage),
so scoring falls back to strict exact string matching on free-form physics answers — which misses
correct answers that are phrased differently or use different notation.

**Next steps once all evals finish:**
1. Re-score physics datasets for all 3 runs using the standalone scorer (`dcvlr_standalone_scorer.py`) with qwen3-4b judge run separately (no vllm conflict)
2. Run Eval 3 once Run 2 frees the GPUs

---

## Infrastructure

- **Machine:** 4× NVIDIA H100 80GB HBM3
- **Training env:** `miniconda3/envs/oumi` (oumi 0.7, torch 2.8.0+cu128)
- **Eval env:** `miniconda3/envs/qwen3-eval`
- **Eval framework:** VLMEvalKit (fork: `scwatson4/VLMEvalKit`)
- **Judge:** `qwen3-4b` → `Qwen/Qwen3-4B-Instruct-2507`
- **Inference:** vllm with tensor_parallel_size=4, gpu_memory_utilization=0.7
