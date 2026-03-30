# DCVLR Eval Progress — 2026-03-30

Evaluation of 3 Qwen3-VL-8B fine-tuned checkpoints on:
`VMCBench_DEV · LiveXivTQA · OlympiadBench · Omni3DBench · atomic · electro · mechanics · optics · quantum · statistics`

Judge: `qwen3-4b` (falls back to exact match / SymPy when GPU unavailable alongside vllm)

---

## Run 1 — `qwen3vl_diverse_cluster_seed2025`

**HF checkpoint:** `scw71432/qwen3vl-8b-diverse-cluster-seed2025`
**Training dataset:** `yosubshin/WaltonMultimodalColdStart-diverse-1k-42` (1k samples, seed 2025)

| Benchmark | Status | Score |
|---|---|---|
| VMCBench_DEV | ✅ Complete | **77.0%** overall |
| LiveXivTQA | ⚠️ Inference done, scoring pending | — |
| OlympiadBench | 🔄 Inference in progress (~87%) | — |
| Omni3DBench | ⏳ Queued | — |
| atomic_dataset | ⏳ Queued | — |
| electro_dataset | ⏳ Queued | — |
| mechanics_dataset | ⏳ Queued | — |
| optics_dataset | ⏳ Queued | — |
| quantum_dataset | ⏳ Queued | — |
| statistics_dataset | ⏳ Queued | — |

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

---

## Run 2 — `qwen3vl_exclude_geo_bio_stats_seed2025`

**HF checkpoint:** `scw71432/qwen3vl-8b-exclude-geo-bio-stats-seed2025`
**Training dataset:** `yosubshin/oumi-walton-exclude-geometry-biology-statistics` (seed 2025)

| Benchmark | Status | Score |
|---|---|---|
| All benchmarks | ⏳ Not started | — |

---

## Run 3 — `qwen3vl_hard_exclude_geo_bio_stats_seed2025`

**HF checkpoint:** `scw71432/qwen3vl-8b-hard-exclude-geo-bio-stats-seed2025`
**Training dataset:** `yosubshin/walton-hard-exclude-geometry-biology-statistics-1k-1` (seed 2025)

| Benchmark | Status | Score |
|---|---|---|
| All benchmarks | ⏳ Not started | — |

---

## Notes

- **OlympiadBench** uses rule-based `MathJudger` (no LLM judge needed)
- **Omni3DBench** uses rule-based `Omni3DBench_acc` (no LLM judge needed)
- **Physics datasets** (atomic/electro/mechanics/optics/quantum/statistics) use exact match + SymPy fallback when LLM judge is unavailable
- **LiveXivTQA** scoring failed with CUDA OOM — vllm inference model holds ~67 GiB across all 4 GPUs, leaving no room for the qwen3-4b judge loaded via HFChatModel. Predictions are saved; will be re-scored. Fix applied to `image_mcq.py` (try/except fallback to exact matching).

## Bug Fixes Applied

| File | Fix |
|---|---|
| `vlmeval/utils/model_detection.py` | Added `Qwen3VLChat` to vllm-compatible classes so custom Qwen3-VL checkpoints get `use_vllm=True` |
| `vlmeval/vlm/qwen3_vl/model.py` | `is_moe_model()` now reads `config.json` for `num_experts` instead of relying solely on path name heuristics (paths without `4B`/`8B` were incorrectly flagged as MoE) |
| `vlmeval/dataset/image_mcq.py` | Added try/except around `build_judge()` in the `'qwen' in model` branch — falls back to exact matching on failure (OOM or otherwise) |
