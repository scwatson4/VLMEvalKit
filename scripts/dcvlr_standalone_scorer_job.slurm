#!/bin/bash
#SBATCH --job-name=VLMEvalKit
#SBATCH --partition=kill-shared
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint="hopper"
#SBATCH --gres=gpu:1

set -euo pipefail

module purge >/dev/null 2>&1 || true

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
SHARED_CACHE_ROOT="${KOA_REMOTE_ROOT:-${HOME}/koa-remote}/cache"
CACHE_ROOT="${KOA_CACHE_ROOT:-${SHARED_CACHE_ROOT}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${RESULTS_DIR:-${JOB_DIR}/results}"
mkdir -p "${REPO_DIR}" "${RESULTS_DIR}" "${CACHE_ROOT}"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${XDG_CACHE_HOME}/hf"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export TMPDIR="${XDG_CACHE_HOME}/tmp"
export APPTAINER_CACHEDIR="${XDG_CACHE_HOME}/apptainer"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${APPTAINER_CACHEDIR}"


if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "KOA_ML_CODE_ROOT=${KOA_ML_CODE_ROOT:-unset}"
echo "KOA_RUN_DIR=${KOA_RUN_DIR:-unset}"
echo "KOA_RUN_METADATA_DIR=${KOA_RUN_METADATA_DIR:-unset}"
echo "KOA_SHARED_ENV=${KOA_SHARED_ENV:-unset}"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo

if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ MIG- ]]; then
  echo "Detected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "This job was scheduled onto a MIG slice, which vLLM cannot handle yet. Exiting early so the job can be resubmitted to a full GPU."
  exit 2
fi

echo "==== GPU Memory Detection ===="
# Query first GPU's name and total VRAM (in MB)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)

echo "Detected GPU: ${GPU_NAME} with ${GPU_MEM} MB VRAM"

# Default settings
BATCH_SIZE=4
DTYPE="bfloat16"

# Adjust batch size and other configs based on VRAM capacity
if (( GPU_MEM < 30000 )); then
    echo "⚠️ GPU has <30GB VRAM — too small for Qwen2.5-VL-7B vLLM."
    exit 1
elif (( GPU_MEM < 50000 )); then
    BATCH_SIZE=8
elif (( GPU_MEM < 80000 )); then
    BATCH_SIZE=16
else
    BATCH_SIZE=50
fi

echo "Using batch size: ${BATCH_SIZE}"

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

source "scripts/setup_env.sh"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_MEMORY_UTILIZATION=0.85

# Allow overriding the model path while keeping a filesystem-safe label for outputs.
CUSTOM_MODEL="${CUSTOM_MODEL:-/home/yosubs/koa_scratch/oumi/output/qwen2_5_vl_7b_walton_random_5000_3}"
CUSTOM_MODEL_SAFE="${CUSTOM_MODEL//\//_}"

srun uv run scripts/dcvlr_standalone_scorer.py \
      --benchmarks VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \
      --input-dir "/home/yosubs/koa_scratch/VLMEvalKit/outputs/${CUSTOM_MODEL_SAFE}" \
      --llm-backend qwen --model qwen3-4b \
      --num-workers 8 \
      --qwen-judge-batch-size 128 \
      --verbose