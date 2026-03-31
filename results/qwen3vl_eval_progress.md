# DCVLR Eval Progress — 2026-03-30

Evaluation of 3 Qwen3-VL-8B fine-tuned checkpoints on:
`VMCBench_DEV · LiveXivTQA · OlympiadBench · Omni3DBench · atomic · electro · mechanics · optics · quantum · statistics`

Judge: `qwen3-4b` (falls back to exact match / SymPy when GPU unavailable alongside vllm)

---

## Run 1 — `qwen3vl_diverse_cluster_seed2025` ✅ Complete

**HF checkpoint:** `scw71432/qwen3vl-8b-diverse-cluster-seed2025`
**Training dataset:** `yosubshin/WaltonMultimodalColdStart-diverse-1k-42` (1k samples, seed 2025)

| Benchmark | Status | Score | Notes |
|---|---|---|---|
| VMCBench_DEV | ✅ Complete | **77.0%** overall | Rule-based scorer |
| LiveXivTQA | ✅ Complete | **77.27%** | Regex extraction of `\boxed{X}` (judge OOM'd alongside vllm) |
| OlympiadBench | ✅ Complete | **15.05%** AVG | math 21.22%, physics 3.32% |
| Omni3DBench | ✅ Complete | See below | Rule-based |
| atomic_dataset | ✅ Complete | **2.5%** ⚠️ | Exact match fallback — needs re-scoring |
| electro_dataset | ✅ Complete | **1.24%** ⚠️ | Same |
| mechanics_dataset | ✅ Complete | **0.0%** ⚠️ | Same |
| optics_dataset | ✅ Complete | **1.27%** ⚠️ | Same |
| quantum_dataset | ✅ Complete | **1.27%** ⚠️ | Same |
| statistics_dataset | ✅ Complete | **1.67%** ⚠️ | Same |

### VMCBench_DEV breakdown

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

### OlympiadBench breakdown

| Subset | Score |
|---|---|
| OE_MM_maths_en_COMP | 23.33% |
| OE_MM_maths_zh_CEE | 14.19% |
| OE_MM_maths_zh_COMP | 12.50% |
| OE_MM_physics_en_COMP | 3.51% |
| OE_MM_physics_zh_CEE | 3.03% |
| OE_TO_maths_en_COMP | 34.81% |
| OE_TO_maths_zh_CEE | 26.29% |
| OE_TO_maths_zh_COMP | 16.67% |
| OE_TO_physics_en_COMP | 5.08% |
| OE_TO_physics_zh_CEE | 2.61% |
| zh_maths | 18.59% |
| zh_physics | 3.00% |
| en_maths | 32.73% |
| en_physics | 4.05% |
| maths | 21.22% |
| physics | 3.32% |
| **AVG** | **15.05%** |

### Omni3DBench breakdown

| Metric | Score |
|---|---|
| Yes/No Accuracy | 52.0% |
| Multiple Choice Accuracy | 62.8% |
| Numeric (count) Accuracy | 17.1% |
| Numeric (other) Mean Relative Accuracy | 29.1% |

---

## Run 2 — `qwen3vl_exclude_geo_bio_stats_seed2025` 🔄 In Progress

**HF checkpoint:** `scw71432/qwen3vl-8b-exclude-geo-bio-stats-seed2025`
**Training dataset:** `yosubshin/oumi-walton-exclude-geometry-biology-statistics` (seed 2025)

| Benchmark | Status | Score | Notes |
|---|---|---|---|
| VMCBench_DEV | ✅ Complete | **74.9%** overall | |
| LiveXivTQA | ✅ Complete | **75.87%** | qwen3-4b judge |
| OlympiadBench | 🔄 In progress | — | Last write: ~12:03 UTC |
| Omni3DBench | ⏳ Queued | — | |
| atomic_dataset | ⏳ Queued | — | |
| electro_dataset | ⏳ Queued | — | |
| mechanics_dataset | ⏳ Queued | — | |
| optics_dataset | ⏳ Queued | — | |
| quantum_dataset | ⏳ Queued | — | |
| statistics_dataset | ⏳ Queued | — | |

### VMCBench_DEV breakdown

| Category | Score |
|---|---|
| Overall | 74.9% |
| General | 83.14% |
| Reasoning | 54.67% |
| OCR | 93.0% |
| Doc & Chart | 80.4% |
| SEEDBench | 82.0% |
| MMStar | 78.0% |
| A-OKVQA | 84.0% |
| VizWiz | 82.0% |
| MMVet | 74.0% |
| VQAv2 | 94.0% |
| OKVQA | 88.0% |
| MMMU | 46.0% |
| MathVista | 52.0% |
| ScienceQA | 80.0% |
| RealWorldQA | 46.0% |
| GQA | 72.0% |
| MathVision | 32.0% |
| TextVQA | 92.0% |
| OCRVQA | 94.0% |
| AI2D | 62.0% |
| ChartQA | 86.0% |
| DocVQA | 100.0% |
| InfoVQA | 78.0% |
| TableVQABench | 76.0% |

---

## Run 3 — `qwen3vl_hard_exclude_geo_bio_stats_seed2025` ⏳ Not started

**HF checkpoint:** `scw71432/qwen3vl-8b-hard-exclude-geo-bio-stats-seed2025` ✅ Uploaded
**Training dataset:** `yosubshin/walton-hard-exclude-geometry-biology-statistics-1k-1` (seed 2025)

Blocked until Run 2 frees the GPUs.

---

## Notes

- **OlympiadBench** uses rule-based `MathJudger` (no LLM judge needed)
- **Omni3DBench** uses rule-based `Omni3DBench_acc` (no LLM judge needed)
- **Physics datasets** (atomic/electro/mechanics/optics/quantum/statistics) use exact match + SymPy fallback when LLM judge is unavailable alongside vllm — scores are near-zero and need re-scoring
- **LiveXivTQA Run 1** scored via regex extraction of `\boxed{X}` letter from predictions (judge OOM'd); Run 2 judge succeeded

## Bug Fixes Applied

| File | Fix |
|---|---|
| `vlmeval/utils/model_detection.py` | Added `Qwen3VLChat` to vllm-compatible classes so custom Qwen3-VL checkpoints get `use_vllm=True` |
| `vlmeval/vlm/qwen3_vl/model.py` | `is_moe_model()` now reads `config.json` for `num_experts` instead of relying solely on path name heuristics (paths without `4B`/`8B` were incorrectly flagged as MoE) |
| `vlmeval/dataset/image_mcq.py` | Added try/except around `build_judge()` in the `'qwen' in model` branch — falls back to exact matching on failure (OOM or otherwise) |
