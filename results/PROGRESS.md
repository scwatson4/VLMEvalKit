# Molmo-7B SFT Pipeline — Full Progress Summary
Last updated: 2026-03-30

---

## 1. Environment Setup ✅
- Conda env: `vlm-eval` (Python 3.10) at `/media/volume/ICML_TESTING_NEW_MODELS/miniconda3/envs/vlm-eval`
- Key packages: torch 2.6.0+cu124, flash-attn 2.6.3, flashinfer, vLLM, VLMEvalKit
- Framework: YosubShin/VLMEvalKit fork (scwatson4/VLMEvalKit)

---

## 2. Baseline Eval ✅
- **Model:** `oumi-ai/Molmo-7B-D-0924` (untuned)
- **Dataset:** WaltonMultimodalColdStart (10,000 questions)
- **Judge:** `meta-llama/Llama-3.1-8B-Instruct` (alias: llama31-8b)
- **Score: 78.03%** (7,803 / 10,000 correct)

---

## 3. SFT Run 1 — oumi-walton-exclude-geometry ✅
- **Training dataset:** `yosubshin/oumi-walton-exclude-geometry`
- **Config:** TRL_SFT, Adafactor, cosine LR, 1 epoch, 1000 steps
- **Training stats:** loss=0.7290, ~535s, ~1980 tokens/s
- **Checkpoint:** `scw71432/Molmo-7B-D-WaltonSFT` (HuggingFace)
- **Eval dataset:** WaltonMultimodalColdStart (same as baseline)
- **Score: 76.96%** (-1.07% vs baseline)
- **Note:** Likely regressed because training excluded geometry, but eval includes it

---

## 4. SFT Run 2 — Three Variants

All 3 models trained ✅, uploaded to HuggingFace ✅, eval running.

### Dataset 1: diverse_cluster
- **HF Repo:** `scw71432/molmo-7b-diverse-cluster-seed2025`
- **Training dataset:** `yosubshin/WaltonMultimodalColdStart-diverse-1k-42` (1,000 samples)
- **Training stats:** 1000 steps, loss=0.607, ~535s, ~2162 tokens/s

### Dataset 4: exclude_geo_bio_stats
- **HF Repo:** `scw71432/molmo-7b-exclude-geo-bio-stats-seed2025`
- **Training dataset:** `yosubshin/oumi-walton-exclude-geometry-biology-statistics`
- **Training stats:** 1000 steps, loss=0.681, ~508s, ~1928 tokens/s

### Dataset 9: hard_exclude_geo_bio_stats
- **HF Repo:** `scw71432/molmo-7b-hard-exclude-geo-bio-stats-seed2025`
- **Training dataset:** `yosubshin/walton-hard-exclude-geometry-biology-statistics-1k-1` (945 samples)
- **Training stats:** 945 steps, loss=0.662, ~508s, ~2037 tokens/s

---

## 5. Eval Benchmarks (10 datasets)

| Benchmark | # Samples | Notes |
|-----------|-----------|-------|
| VMCBench_DEV | 1,000 | Multi-category VQA |
| LiveXivTQA | 7,913 | Live arXiv QA |
| OlympiadBench | 5,929 | Math Olympiad — requires antlr4 ≥4.11 for LaTeX |
| Omni3DBench | 501 | 3D visual understanding |
| atomic_dataset | 200 | Physics (atomic) — SymPy fallback (no OpenAI key) |
| electro_dataset | 242 | Physics (electromagnetism) |
| mechanics_dataset | 221 | Physics (mechanics) |
| optics_dataset | 158 | Physics (optics) |
| quantum_dataset | 236 | Physics (quantum) |
| statistics_dataset | 240 | Physics (statistics) |
| **Total** | **16,640** | |

---

## 6. Eval Results — diverse_cluster

Eval started: 2026-03-29 22:00 UTC
Current time: 2026-03-30 20:10 UTC (22h running)

| Benchmark | Status | Score / Notes |
|-----------|--------|---------------|
| VMCBench_DEV | ✅ Complete | **55.9%** overall (see breakdown below) |
| Omni3DBench | ✅ Complete | Yes/No=53.3%, MC=46.5%, Count=7.1%, Other=15.9% |
| atomic_dataset | ✅ Complete | **0.5%** ⚠️ SymPy-only (no OpenAI key — severely underestimated) |
| LiveXivTQA | ❌ Eval error | CUDA OOM during LLM judge — inference saved (7913 preds) |
| OlympiadBench | ❌ Eval error | antlr4 4.9.3 installed, requires 4.11 — SymPy fallback running |
| electro_dataset | 🔄 Judging | qwen3-4b judge running (~7h) |
| mechanics_dataset | ⏳ Pending | — |
| optics_dataset | ⏳ Pending | — |
| quantum_dataset | ⏳ Pending | — |
| statistics_dataset | ⏳ Pending | — |

### VMCBench_DEV Breakdown (diverse_cluster, 55.9% overall)
| Category | Score |
|----------|-------|
| Overall | 55.9% |
| General | 66.6% |
| Reasoning | 40.7% |
| OCR | 77.0% |
| Doc & Chart | 50.8% |
| SEEDBench | 80.0% |
| MMStar | 58.0% |
| A-OKVQA | 74.0% |
| VizWiz | 74.0% |
| MMVet | 42.0% |
| VQAv2 | 68.0% |
| OKVQA | 70.0% |
| MMMU | 32.0% |
| MathVista | 38.0% |
| ScienceQA | 58.0% |
| RealWorldQA | 34.0% |
| GQA | 64.0% |
| MathVision | 18.0% |
| TextVQA | 70.0% |
| OCRVQA | 84.0% |
| AI2D | 44.0% |
| ChartQA | 42.0% |
| DocVQA | 62.0% |
| InfoVQA | 62.0% |
| TableVQABench | 44.0% |

---

## 7. Eval Results — exclude_geo_bio_stats & hard_exclude_geo_bio_stats

Both queued — will start automatically after diverse_cluster eval completes.

| Model | Status |
|-------|--------|
| exclude_geo_bio_stats | ⏳ Queued |
| hard_exclude_geo_bio_stats | ⏳ Queued |

---

## 8. All Completed Scores Summary

| Run | Benchmark | Score |
|-----|-----------|-------|
| Baseline (untuned) | WaltonMultimodalColdStart | **78.03%** |
| SFT Run 1 (exclude-geometry) | WaltonMultimodalColdStart | **76.96%** (-1.07%) |
| diverse_cluster | VMCBench_DEV | **55.9%** |
| diverse_cluster | Omni3DBench | Yes/No=53.3%, MC=46.5% |
| diverse_cluster | atomic_dataset | 0.5% ⚠️ (no OpenAI key) |
| exclude_geo_bio_stats | All benchmarks | ⏳ Pending |
| hard_exclude_geo_bio_stats | All benchmarks | ⏳ Pending |

---

## 9. Known Issues & Fixes

| Issue | Status | Fix/Note |
|-------|--------|----------|
| vLLM mm_input_cache KeyError (inference) | ✅ Fixed | `disable_mm_preprocessor_cache=True` in molmo.py |
| vLLM mm_input_cache KeyError (judging) | ✅ Fixed | `disable_mm_preprocessor_cache=True` in walton_multimodal.py |
| Judge path corruption (`/` in model name) | ✅ Fixed | `llama31-8b` shortname mapping added |
| OOM when judge loads alongside inference | ✅ Fixed | Run judging after GPU fully freed |
| torchrun port conflict (EADDRINUSE 29500) | ✅ Fixed | Use `oumi train` CLI directly |
| `use_torchdata: true` crash | ✅ Fixed | `use_torchdata: false` in all configs |
| `torch_dtype_str: bfloat16` dtype mismatch | ✅ Fixed | `float32` + `mixed_precision_dtype: BF16` |
| LiveXivTQA judge CUDA OOM | ❌ Open | Inference complete, judge OOM — needs retry with smaller batch |
| OlympiadBench antlr4 version mismatch | ❌ Open | Requires antlr4 ≥4.11, installed 4.9.3 |
| Physics eval no OpenAI key | ❌ Open | SymPy-only fallback severely underestimates accuracy |

---

## 10. GitHub Repo — scwatson4/VLMEvalKit

Key commits:
- `llama31-8b` judge shortname mapping
- `disable_mm_preprocessor_cache=True` for inference and judging
- Molmo SFT training configs (`configs/dcvlr/`)
- `PROGRESS.md`, `SCORES_BASELINE.md`, `SCORES_FINETUNED.md`, `SCORES_SUMMARY.md`
- Baseline & finetuned judge xlsx in `results/`

---

*Pipeline running autonomously. Results table will print when all 3 evals complete.*
