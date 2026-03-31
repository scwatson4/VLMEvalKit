# Molmo-7B SFT — Results Summary
Last updated: 2026-03-31

> **Pipeline status:** diverse_cluster eval running (~33h), exclude/hard_exclude queued.
> Results below reflect all completed scoring as of this update.

---

## Overview

| Run | Training Dataset | HF Checkpoint | WaltonMC Score |
|-----|-----------------|---------------|----------------|
| Baseline (untuned) | — | `oumi-ai/Molmo-7B-D-0924` | **78.03%** |
| SFT Run 1 | oumi-walton-exclude-geometry | `scw71432/Molmo-7B-D-WaltonSFT` | **76.96%** (−1.07%) |
| SFT Run 2a: diverse_cluster | WaltonMC-diverse-1k-42 | `scw71432/molmo-7b-diverse-cluster-seed2025` | pending |
| SFT Run 2b: exclude_geo_bio_stats | walton-exclude-geo-bio-stats | `scw71432/molmo-7b-exclude-geo-bio-stats-seed2025` | pending |
| SFT Run 2c: hard_exclude_geo_bio_stats | walton-hard-exclude-geo-bio-stats-1k | `scw71432/molmo-7b-hard-exclude-geo-bio-stats-seed2025` | pending |

---

## SFT Training Stats (Run 2)

| Model | Dataset | Samples | Steps | Loss | Time | Tokens/s |
|-------|---------|---------|-------|------|------|----------|
| diverse_cluster | WaltonMC-diverse-1k-42 | 1,000 | 1000 | 0.607 | ~535s | ~2162 |
| exclude_geo_bio_stats | walton-exclude-geo-bio-stats | full | 1000 | 0.681 | ~508s | ~1928 |
| hard_exclude_geo_bio_stats | walton-hard-exclude-geo-bio-stats-1k | 945 | 945 | 0.662 | ~508s | ~2037 |

Config: TRL_SFT, Adafactor, cosine LR 2e-5, 1 epoch, BF16 mixed precision, single GPU

---

## Benchmark Eval — diverse_cluster (partial, in progress)

Eval started: 2026-03-29 22:00 UTC

### VMCBench_DEV — ✅ Complete

| Category | Score (%) |
|----------|-----------|
| **Overall** | **55.90** |
| General | 66.57 |
| Reasoning | 40.67 |
| OCR | 77.00 |
| Doc & Chart | 50.80 |
| SEEDBench | 80.00 |
| MMStar | 58.00 |
| A-OKVQA | 74.00 |
| VizWiz | 74.00 |
| MMVet | 42.00 |
| VQAv2 | 68.00 |
| OKVQA | 70.00 |
| MMMU | 32.00 |
| MathVista | 38.00 |
| ScienceQA | 58.00 |
| RealWorldQA | 34.00 |
| GQA | 64.00 |
| MathVision | 18.00 |
| TextVQA | 70.00 |
| OCRVQA | 84.00 |
| AI2D | 44.00 |
| ChartQA | 42.00 |
| DocVQA | 62.00 |
| InfoVQA | 62.00 |
| TableVQABench | 44.00 |

### Omni3DBench — ✅ Complete

| Metric | Score |
|--------|-------|
| Yes/No Accuracy | 53.33% |
| Multiple Choice Accuracy | 46.51% |
| Numeric (count) Accuracy | 7.14% |
| Numeric (other) Mean Relative Accuracy | 15.89% |

### LiveXivTQA — ❌ Eval Error

Inference complete (7,913 predictions saved). Scoring failed: CUDA OOM when
`HFChatModel` judge tried to load onto GPU already occupied by vLLM inference
model (~57 GB used). Needs two-phase run or vLLM memory release before judging.

### OlympiadBench — ❌ Eval Error

Inference complete (6,765 predictions saved). Scoring failed: `MathJudger`
requires `antlr4-python3-runtime==4.11`; installed version is 4.9.3.
Fix: `pip install "antlr4-python3-runtime==4.11"` in vlm-eval env.

### Physics Datasets — ⚠️ Severely Underestimated

Physics datasets use a `Physics_yale` evaluator with OpenAI API for LLM-based
answer equivalence. Without an OpenAI key, it falls back to SymPy exact
matching, which almost never matches model outputs written in natural language.
**These scores are not meaningful without an OpenAI key or a patched evaluator.**

| Dataset | Samples | Score (SymPy-only) | Note |
|---------|---------|-------------------|------|
| atomic_dataset | 200 | **0.5%** (1/200) | SymPy fallback only |
| electro_dataset | 242 | pending | qwen3-4b judge stalled |
| mechanics_dataset | 221 | pending | — |
| optics_dataset | 158 | pending | — |
| quantum_dataset | 236 | pending | — |
| statistics_dataset | 240 | pending | — |

---

## Benchmark Eval — exclude_geo_bio_stats & hard_exclude_geo_bio_stats

⏳ Queued — will start automatically when diverse_cluster eval finishes.

---

## Known Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| LiveXivTQA judge CUDA OOM | vLLM holds GPU memory; HFChatModel tries to load judge on same GPU | Two-phase eval (inference then judging in separate process) |
| OlympiadBench antlr4 error | antlr4 4.9.3 installed, 4.11 required | `pip install antlr4-python3-runtime==4.11` |
| Physics scores ~0% | No OpenAI API key → SymPy exact-match fallback | Provide OpenAI key or patch evaluator to use qwen3-4b |
| electro/physics judging stalled | OlympiadBench SymPy CPU LaTeX parse blocks (6,765 items) | Kill and rerun after fixing antlr4 |
