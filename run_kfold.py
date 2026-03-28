#!/usr/bin/env python3
"""
K-fold inference script for VLMEvalKit.

This script runs inference k times for each prompt to assess question difficulty
and reliability. Each prompt gets k different responses which are evaluated
individually, then aggregated with a verdict_sum.

Usage:
    python run_kfold.py --data WaltonMultimodalReasoning --model qwen2_vl --k 8
"""

import os
import sys
import torch
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
import os.path as osp
from uuid import uuid4
import subprocess
import gc


# Setup multi-GPU environment (similar to run.py)
def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if CUDA_VISIBLE_DEVICES != "":
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(",")]
        return gpu_list
    try:
        ps = subprocess.Popen(("nvidia-smi", "--list-gpus"), stdout=subprocess.PIPE)
        output = subprocess.check_output(("wc", "-l"), stdin=ps.stdout)
        return list(range(int(output)))
    except:
        # no nvidia-smi, maybe a mac/ROCm?
        return []


# Setup distributed environment variables
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

GPU_LIST = get_gpu_list()
if LOCAL_WORLD_SIZE > 1 and len(GPU_LIST):
    NGPU = len(GPU_LIST)
    assert (
        NGPU >= LOCAL_WORLD_SIZE
    ), "The number of processes should be less than or equal to the number of GPUs"
    GPU_PER_PROC = NGPU // LOCAL_WORLD_SIZE
    DEVICE_START_IDX = GPU_PER_PROC * LOCAL_RANK
    CUDA_VISIBLE_DEVICES = [
        str(i) for i in GPU_LIST[DEVICE_START_IDX : DEVICE_START_IDX + GPU_PER_PROC]
    ]
    CUDA_VISIBLE_DEVICES = ",".join(CUDA_VISIBLE_DEVICES)
    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    print(
        f"RANK: {RANK}, LOCAL_RANK: {LOCAL_RANK}, WORLD_SIZE: {WORLD_SIZE}, "
        f"LOCAL_WORLD_SIZE: {LOCAL_WORLD_SIZE}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}"
    )

# VLMEvalKit imports
from vlmeval import *
from vlmeval.dataset import build_dataset
from vlmeval.config import supported_VLM
from vlmeval.smp import *


def parse_args():
    """Parse command line arguments for k-fold inference."""
    help_msg = """
K-fold inference script for VLMEvalKit.

This script runs inference k times for each prompt to assess question difficulty
and reliability. Each prompt gets k different responses which are evaluated
individually, then aggregated with a verdict_sum.

Usage:
    python run_kfold.py --data WaltonMultimodalReasoning --model qwen2_vl --k 8
"""
    parser = argparse.ArgumentParser(
        description=help_msg, formatter_class=argparse.RawTextHelpFormatter
    )

    # Essential Args
    parser.add_argument("--data", type=str, nargs="+", help="Names of Datasets")
    parser.add_argument("--model", type=str, help="Name of Model")

    # Work Dir
    parser.add_argument(
        "--work-dir", type=str, default="./outputs", help="select the output directory"
    )

    # API Kwargs
    parser.add_argument("--nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument(
        "--api-nproc",
        type=int,
        default=4,
        help="Parallel API calling (alias for nproc)",
    )
    parser.add_argument(
        "--retry", type=int, default=None, help="retry numbers for API VLMs"
    )
    parser.add_argument(
        "--judge", type=str, default="gpt-4o-mini", help="Judge model name"
    )

    # Model and Generation Settings
    parser.add_argument(
        "--pass-custom-model",
        type=str,
        default=None,
        help="Path to a HuggingFace repository or local model directory",
    )
    # VLLM is required for k-fold; do not expose a toggle
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for VLLM inference"
    )
    parser.add_argument(
        "--force-sequential-multimodal-vllm",
        action="store_true",
        help=(
            "Force sequential per-prompt processing for multimodal Qwen vLLM runs "
            "when the vLLM build cannot disable the multimodal preprocessor cache"
        ),
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=None,
        help="Batch size for judge evaluation (Walton/vLLM judge path)",
    )
    parser.add_argument(
        "--walton-judge-impl",
        choices=["default", "multistage"],
        default="default",
        help="WaltonMultimodalReasoning judge implementation to use during evaluation",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens for generation",
    )
    # Note: When using VLLM, we will set SamplingParams.n = k

    # Resume/Reuse Settings
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse existing prediction files"
    )

    # Logging Utils
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-warning", action="store_true", help="Disable warnings")
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Remove 'image' column from each row before building prompts",
    )

    # K-fold specific arguments
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of times to run inference per prompt (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling variation (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling (default: 0.9)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed for reproducibility (actual seed = base + k_iteration)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dataset rows to process for debugging",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.k < 2:
        print("ERROR: K must be at least 2 for meaningful k-fold inference")
        sys.exit(1)

    if args.limit is not None and args.limit < 1:
        print("ERROR: --limit must be at least 1")
        sys.exit(1)

    if not args.data:
        print("ERROR: --data is required")
        sys.exit(1)

    if not args.model and not args.pass_custom_model:
        print("ERROR: Either --model or --pass-custom-model must be specified")
        sys.exit(1)

    return args


def build_model(model_name, **kwargs):
    """Build and return the model for inference."""
    logger = get_logger("RUN")

    if model_name in supported_VLM:
        model_cls = supported_VLM[model_name]
        model = model_cls(**kwargs)

        # Ensure model has temperature settings
        if hasattr(model, "temperature"):
            model.temperature = kwargs.get("temperature", 0.7)
        if hasattr(model, "top_p"):
            model.top_p = kwargs.get("top_p", 0.9)

        return model
    else:
        logger.error(f"Model {model_name} not supported")
        raise ValueError(f"Model {model_name} not supported")


def infer_kfold_batch(
    model,
    dataset,
    k=8,
    temperature=0.7,
    top_p=0.9,
    seed_base=42,
    work_dir="./outputs",
    verbose=False,
    reuse=False,
    batch_size=None,
    model_name=None,
    skip_image=False,
):
    """
    Run k-fold inference with batch processing optimization.

    When batch_size is provided and the model supports batch processing,
    we can run multiple k-iterations across different prompts in a single batch.
    For example, with batch_size=32 and k=8, we can process 4 prompts simultaneously
    (each prompt gets 8 iterations = 32 total).

    Args:
        model: The VLM model to use
        dataset: The dataset to evaluate
        k: Number of inference iterations per prompt
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        seed_base: Base seed for reproducibility
        work_dir: Directory to save outputs
        verbose: Whether to print verbose output
        reuse: Whether to reuse existing results
        batch_size: Batch size for inference (enables batch k-fold if set)

    Returns:
        dict: Results with k predictions per index
    """
    logger = get_logger("RUN")

    # Enforce specialized multi-n batch API
    if not hasattr(model, "generate_batch_with_n"):
        raise RuntimeError("Model must implement generate_batch_with_n for k-fold")

    # Check if batch processing is available and should be used
    if (
        batch_size
        and hasattr(model, "supports_batch_processing")
        and model.supports_batch_processing()
    ):
        logger.info(f"Using batch k-fold inference with batch_size={batch_size}, k={k}")

        # With VLLM n>1 we generate n candidates per prompt in one call
        # so we only need 1 iteration per prompt
        prompts_per_batch = batch_size
        if prompts_per_batch < 1:
            logger.warning(
                f"Batch size {batch_size} is smaller than k={k}, falling back to regular k-fold"
            )
            return infer_kfold(
                model,
                dataset,
                k,
                temperature,
                top_p,
                seed_base,
                work_dir,
                verbose,
                reuse,
            )

        logger.info(
            f"Processing {prompts_per_batch} prompts per batch ({prompts_per_batch} * {k} = {batch_size})"
        )
        return _infer_kfold_batched(
            model,
            dataset,
            k,
            prompts_per_batch,
            batch_size,
            temperature,
            top_p,
            seed_base,
            work_dir,
            verbose,
            reuse,
            model_name=model_name,
            skip_image=skip_image,
        )
    else:
        # Fall back to regular k-fold
        if batch_size:
            logger.info(
                "Model does not support batch processing, using sequential k-fold"
            )
        return infer_kfold(
            model, dataset, k, temperature, top_p, seed_base, work_dir, verbose, reuse
        )


def get_model_name(model):
    """Get consistent model name for file naming"""
    if hasattr(model, "model_name") and model.model_name:
        return model.model_name.replace("/", "_").replace("\\", "_")
    else:
        return model.__class__.__name__ if hasattr(model, "__class__") else str(model)


def _infer_kfold_batched(
    model,
    dataset,
    k,
    prompts_per_batch,
    batch_size,
    temperature,
    top_p,
    seed_base,
    work_dir,
    verbose,
    reuse,
    model_name=None,
    skip_image=False,
):
    """
    Internal function for batched k-fold inference using existing batch processing infrastructure.
    Supports multi-GPU by splitting dataset across ranks.
    """
    from vlmeval.utils.batch_processing import BatchCollector, BatchProcessor

    logger = get_logger("RUN")
    dataset_name = dataset.dataset_name
    # Use provided model_name directly; assume it's consistent across the run
    assert model_name is not None, "model_name must be provided from main()"

    # Get rank and world size for distributed processing
    rank, world_size = get_rank_and_world_size()

    logger.info(
        f"Starting batched inference with {k} predictions per prompt, batch_size={batch_size}"
    )
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    logger.info(f"Temperature: {temperature}, Top-p: {top_p}")
    if skip_image:
        logger.info("skip-image enabled: dropping 'image' column before prompt build")
    logger.info(f"Processing up to {prompts_per_batch} prompts per batch")
    if world_size > 1:
        logger.info(f"Distributed inference: Rank {rank}/{world_size}")

    # Prepare output file (rank-specific for multi-GPU)
    os.makedirs(work_dir, exist_ok=True)
    if world_size > 1:
        output_file = osp.join(
            work_dir, f"{model_name}_{dataset_name}_k{k}_rank{rank}.pkl"
        )
    else:
        output_file = osp.join(work_dir, f"{model_name}_{dataset_name}_k{k}.pkl")

    # Load existing results if reusing
    if osp.exists(output_file) and reuse:
        logger.info(f"Reusing existing results from {output_file}")
        results = load(output_file)

        # Check if all complete
        data = dataset.data
        all_complete = all(
            results.get(row["index"], {}).get("predictions", None) is not None
            and len(results.get(row["index"], {}).get("predictions", [])) == k
            for _, row in data.iterrows()
        )
        if all_complete:
            logger.info("All results are complete, skipping inference")
            return results
    elif osp.exists(output_file):
        if verbose:
            logger.info(f"Loading existing results from {output_file} for resumption")
        results = load(output_file)
    else:
        results = {}

    # Get dataset data and split by rank (similar to infer_data in inference.py)
    data = dataset.data
    total_items_global = len(data)

    # Split data across ranks for distributed processing
    if world_size > 1:
        # Each rank processes different indices
        sheet_indices = list(range(rank, len(data), world_size))
        data = data.iloc[sheet_indices]
        total_items = len(data)
        logger.info(
            f"Rank {rank} processing {total_items}/{total_items_global} items (indices: {sheet_indices[:5]}...)"
        )
    else:
        total_items = total_items_global

    # Initialize batch collector and processor (like in infer_data_batch)
    collector = BatchCollector(
        max_batch_size=batch_size,
        batch_timeout=5.0,
        enable_smart_batching=True,
        verbose=verbose,
    )

    processor = BatchProcessor(model, verbose=verbose)

    # Count items that need processing
    items_to_process = 0
    k_iterations_needed = 0
    for _, row in data.iterrows():
        index = row["index"]
        existing_preds = (
            len(results.get(index, {}).get("predictions", []))
            if index in results
            else 0
        )
        if existing_preds < k:
            items_to_process += 1
            k_iterations_needed += k - existing_preds

    logger.info(f"Items to process: {items_to_process}/{total_items}")
    logger.info(f"Total k-iterations needed: {k_iterations_needed}")

    # Progress bar tracks individual items (prompts)
    model_desc = f"{model_name} K-fold"
    progress_bar = tqdm(
        total=total_items, desc=f"Batch K-fold {model_desc}/{dataset_name}"
    )

    # Update progress for already complete items
    already_complete = total_items - items_to_process
    if already_complete > 0:
        progress_bar.update(already_complete)
        if verbose:
            logger.info(
                f"Reusing {already_complete} complete results from previous run"
            )

    # Process all items
    processed_count = 0
    save_counter = 0

    for _, row in data.iterrows():
        index = row["index"]

        # Skip if already complete for this prompt
        if index in results and len(results[index].get("predictions", [])) >= k:
            continue

        # Initialize result structure if needed
        if index not in results:
            results[index] = {
                "index": index,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "predictions": [],
                "metadata": {"temperatures": [], "seeds": [], "top_p_values": []},
            }

        # Build prompt once for this item
        row_for_prompt = row.copy() if skip_image else row
        if skip_image and ("image" in getattr(row_for_prompt, "index", [])):
            # Remove the image column for ablation
            del row_for_prompt["image"]

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            prompt_struct = model.build_prompt(row_for_prompt, dataset=dataset_name)
        else:
            prompt_struct = dataset.build_prompt(row_for_prompt)

        # Add a single generation request per prompt (VLLM can return n candidates)
        existing_preds = len(results[index]["predictions"])
        if existing_preds < k:
            ready_batch = collector.add_item(
                index,
                prompt_struct,
                dataset_name,
            )

            if ready_batch:
                # Process the ready batch
                if verbose:
                    logger.info(f"Processing batch of {len(ready_batch)} prompts")

                # Use specialized multi-n API to guarantee alignment
                messages = [item.message for item in ready_batch]
                nested = model.generate_batch_with_n(
                    messages, dataset=dataset_name, k=k
                )
                for i, out_list in enumerate(nested):
                    orig_index = int(ready_batch[i].index)
                    if orig_index in results:
                        candidates = (
                            out_list if isinstance(out_list, list) else [out_list]
                        )
                        needed = k - len(results[orig_index]["predictions"])
                        to_add = candidates[:needed]
                        results[orig_index]["predictions"].extend(to_add)
                        results[orig_index]["metadata"]["temperatures"].extend(
                            [temperature] * len(to_add)
                        )
                        results[orig_index]["metadata"]["seeds"].extend(
                            [seed_base] * len(to_add)
                        )
                        results[orig_index]["metadata"]["top_p_values"].extend(
                            [top_p] * len(to_add)
                        )

                save_counter += 1
                # Save periodically
                if save_counter % 5 == 0:
                    dump(results, output_file)
                    if verbose:
                        logger.info(f"Saved checkpoint after {save_counter} batches")

        # Update progress after completing all k iterations for this item
        progress_bar.update(1)
        processed_count += 1

        # Clear GPU cache periodically
        if processed_count % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Process any remaining items in the collector
    final_batches = collector.flush_all()
    for final_batch in final_batches:
        if verbose:
            logger.info(f"Processing final batch of {len(final_batch)} prompts")

        messages = [item.message for item in final_batch]
        nested = model.generate_batch_with_n(messages, dataset=dataset_name, k=k)
        for i, out_list in enumerate(nested):
            orig_index = int(final_batch[i].index)
            if orig_index in results:
                candidates = out_list if isinstance(out_list, list) else [out_list]
                needed = k - len(results[orig_index]["predictions"])
                to_add = candidates[:needed]
                results[orig_index]["predictions"].extend(to_add)
                results[orig_index]["metadata"]["temperatures"].extend(
                    [temperature] * len(to_add)
                )
                results[orig_index]["metadata"]["seeds"].extend(
                    [seed_base] * len(to_add)
                )
                results[orig_index]["metadata"]["top_p_values"].extend(
                    [top_p] * len(to_add)
                )

    progress_bar.close()

    # Final save
    dump(results, output_file)
    logger.info(f"Batched k-fold inference complete. Results saved to {output_file}")

    return results


def infer_kfold(
    model,
    dataset,
    k=8,
    temperature=0.7,
    top_p=0.9,
    seed_base=42,
    work_dir="./outputs",
    verbose=False,
    reuse=False,
):
    """
    Run k-fold inference on a dataset.
    Supports multi-GPU by splitting dataset across ranks.

    Args:
        model: The VLM model to use
        dataset: The dataset to evaluate
        k: Number of inference iterations per prompt
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        seed_base: Base seed for reproducibility
        work_dir: Directory to save outputs
        verbose: Whether to print verbose output
        reuse: Whether to reuse existing results

    Returns:
        dict: Results with k predictions per index
    """
    logger = get_logger("RUN")
    dataset_name = dataset.dataset_name
    # Use a clean model name, avoiding path separators
    if hasattr(model, "model_name") and model.model_name:
        model_name = model.model_name.replace("/", "_").replace("\\", "_")
    else:
        model_name = (
            model.__class__.__name__ if hasattr(model, "__class__") else str(model)
        )

    # Get rank and world size for distributed processing
    rank, world_size = get_rank_and_world_size()

    logger.info(f"Starting k-fold inference with k={k}")
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    logger.info(f"Temperature: {temperature}, Top-p: {top_p}")
    if world_size > 1:
        logger.info(f"Distributed inference: Rank {rank}/{world_size}")

    # Prepare output file (rank-specific for multi-GPU)
    os.makedirs(work_dir, exist_ok=True)
    if world_size > 1:
        output_file = osp.join(
            work_dir, f"{model_name}_{dataset_name}_k{k}_rank{rank}.pkl"
        )
    else:
        output_file = osp.join(work_dir, f"{model_name}_{dataset_name}_k{k}.pkl")

    # Load existing results if any (for resumption)
    if osp.exists(output_file) and reuse:
        logger.info(f"Reusing existing results from {output_file}")
        results = load(output_file)

        # If all results are complete, return them
        data = dataset.data
        all_complete = all(
            results.get(row["index"], {}).get("predictions", None) is not None
            and len(results.get(row["index"], {}).get("predictions", [])) == k
            for _, row in data.iterrows()
        )
        if all_complete:
            logger.info("All results are complete, skipping inference")
            return results
    elif osp.exists(output_file):
        if verbose:
            logger.info(f"Loading existing results from {output_file} for resumption")
        results = load(output_file)
    else:
        results = {}

    # Get dataset data and split by rank (similar to infer_data in inference.py)
    data = dataset.data
    total_items_global = len(data)

    # Split data across ranks for distributed processing
    if world_size > 1:
        # Each rank processes different indices
        sheet_indices = list(range(rank, len(data), world_size))
        data = data.iloc[sheet_indices]
        total_items = len(data)
        logger.info(f"Rank {rank} processing {total_items}/{total_items_global} items")
    else:
        total_items = total_items_global

    # Progress bar for overall completion
    pbar = tqdm(
        total=total_items, desc=f"K-fold Inference (k={k}, Rank {rank}/{world_size})"
    )

    for _, row in data.iterrows():
        index = row["index"]

        # Skip if already completed
        if index in results and len(results[index]["predictions"]) == k:
            pbar.update(1)
            continue

        # Initialize result structure for this index
        if index not in results:
            results[index] = {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "predictions": [],
                "temperatures": [],
                "seeds": [],
            }

        # Build prompt once
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            prompt_struct = model.build_prompt(row, dataset=dataset_name)
        else:
            prompt_struct = dataset.build_prompt(row)

        # Generate predictions
        existing_preds = len(results[index]["predictions"])
        use_vllm_multi = (
            hasattr(model, "use_vllm")
            and model.use_vllm
            and hasattr(model, "vllm_n")
            and model.vllm_n
            and model.vllm_n > 1
        )

        if use_vllm_multi:
            # Single VLLM call can return multiple candidates (n=k)
            # Temporarily set model parameters if possible
            original_temp = getattr(model, "temperature", None)
            original_top_p = getattr(model, "top_p", None)
            try:
                if hasattr(model, "temperature"):
                    model.temperature = temperature
                if hasattr(model, "top_p"):
                    model.top_p = top_p

                response = model.generate(message=prompt_struct, dataset=dataset_name)

                # response may be a list of candidates
                if isinstance(response, list):
                    needed = k - existing_preds
                    to_add = response[:needed]
                    results[index]["predictions"].extend(to_add)
                    results[index]["temperatures"].extend([temperature] * len(to_add))
                    results[index]["seeds"].extend([seed_base] * len(to_add))
                else:
                    results[index]["predictions"].append(response)
                    results[index]["temperatures"].append(temperature)
                    results[index]["seeds"].append(seed_base)
            finally:
                if original_temp is not None and hasattr(model, "temperature"):
                    model.temperature = original_temp
                if original_top_p is not None and hasattr(model, "top_p"):
                    model.top_p = original_top_p

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Sequential generation of k predictions
            for k_iter in range(existing_preds, k):
                # Set seed for reproducibility
                current_seed = seed_base + k_iter
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
                torch.manual_seed(current_seed)

                # Temporarily set model parameters if possible
                original_temp = getattr(model, "temperature", None)
                original_top_p = getattr(model, "top_p", None)

                try:
                    if hasattr(model, "temperature"):
                        model.temperature = temperature
                    if hasattr(model, "top_p"):
                        model.top_p = top_p

                    # Generate response
                    response = model.generate(
                        message=prompt_struct, dataset=dataset_name
                    )

                    # Store result
                    results[index]["predictions"].append(response)
                    results[index]["temperatures"].append(temperature)
                    results[index]["seeds"].append(current_seed)

                    if verbose:
                        print(
                            f"Index {index}, Iteration {k_iter+1}/{k}: {response[:100]}..."
                        )

                finally:
                    # Restore original parameters
                    if original_temp is not None and hasattr(model, "temperature"):
                        model.temperature = original_temp
                    if original_top_p is not None and hasattr(model, "top_p"):
                        model.top_p = original_top_p

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save progress periodically
        if (pbar.n + 1) % 10 == 0:
            dump(results, output_file)

        pbar.update(1)

    pbar.close()

    # Final save
    dump(results, output_file)
    logger.info(f"K-fold inference complete. Results saved to {output_file}")

    return results


def convert_to_dataframe(kfold_results, k):
    """
    Convert k-fold results to a DataFrame format suitable for evaluation.

    Args:
        kfold_results: Dictionary with k predictions per index
        k: Number of predictions per index

    Returns:
        pd.DataFrame: DataFrame with separate columns for each prediction
    """
    rows = []

    for index, data in kfold_results.items():
        row = {"index": index, "question": data["question"], "answer": data["answer"]}

        # Add each prediction as a separate column
        for i in range(k):
            if i < len(data["predictions"]):
                row[f"prediction_{i+1}"] = data["predictions"][i]
            else:
                row[f"prediction_{i+1}"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by index for consistency
    df = df.sort_values("index").reset_index(drop=True)

    return df


def evaluate_kfold(
    dataset, df_predictions, k, work_dir="./outputs", judge_model=None, **judge_kwargs
):
    """
    Evaluate k-fold predictions using the dataset's judge.

    Args:
        dataset: The dataset object with evaluate method
        df_predictions: DataFrame with k predictions per row
        k: Number of predictions
        work_dir: Working directory for temporary files
        judge_kwargs: Arguments for the judge model

    Returns:
        pd.DataFrame: DataFrame with verdicts for each prediction
    """
    logger = get_logger("RUN")
    dataset_name = dataset.dataset_name
    logger.info(f"Evaluating k-fold predictions for {dataset_name}")

    results = df_predictions.copy()

    # Build judge model once if not provided
    if judge_model is None:
        use_vllm_judge = judge_kwargs.get("use_vllm_judge", False)
        model_name = judge_kwargs.get("model", "gpt-4o-mini")

        if use_vllm_judge and not model_name.startswith("gpt"):
            logger.info(f"Building VLLM judge model: {model_name}")
            # Keep batch_size in judge_kwargs so dataset.evaluate() can reuse it
            # for its own outer batching instead of falling back to 32.
            judge_model_kwargs = dict(judge_kwargs)
            batch_size = judge_model_kwargs.pop("batch_size", 32)
            judge_model = dataset._build_vllm_judge(
                model_name, batch_size=batch_size, **judge_model_kwargs
            )
        else:
            from vlmeval.dataset.utils.judge_util import build_judge

            judge_model = build_judge(**judge_kwargs)

    try:
        # Check which verdicts already exist for recovery
        existing_verdicts = []
        for i in range(k):
            verdict_col = f"verdict_{i+1}"
            if verdict_col in results.columns and not results[verdict_col].isna().all():
                existing_verdicts.append(i + 1)

        if existing_verdicts:
            logger.info(
                f"Recovery: Found existing verdicts for predictions {existing_verdicts}"
            )
            start_k = len(existing_verdicts) + 1
        else:
            start_k = 1

        # Evaluate each prediction column using the same judge model
        for i in range(start_k - 1, k):
            pred_col = f"prediction_{i+1}"
            verdict_col = f"verdict_{i+1}"
            judge_stage_col = f"judge_stage_{i+1}"
            judge_stage_detail_col = f"judge_stage_detail_{i+1}"

            logger.info(f"Evaluating prediction {i+1}/{k}")

            # Use unique temp file name for this k iteration
            temp_file = osp.join(work_dir, f"temp_eval_k{i+1}_{dataset_name}.xlsx")

            # Create temporary dataframe for evaluation
            temp_df = df_predictions[["index", "question", "answer"]].copy()
            temp_df["prediction"] = df_predictions[pred_col]
            keep_temp_file = False
            eval_df_for_debug = None

            # Save to temporary file (always save for consistency)
            temp_df.to_excel(temp_file, index=False)

            try:
                # Pass the pre-built judge model to avoid re-instantiation
                eval_result = dataset.evaluate(
                    temp_file, judge_model=judge_model, **judge_kwargs
                )

                # Extract verdicts
                loaded_from_file = False
                if isinstance(eval_result, pd.DataFrame):
                    eval_df_for_debug = eval_result
                    # Merge verdict into results
                    if "verdict" in eval_result.columns:
                        results[verdict_col] = eval_result["verdict"].values
                        logger.info(
                            f"Extracted {len(eval_result)} verdicts for prediction {i+1}"
                        )
                    else:
                        if judge_kwargs.get("verbose", False):
                            logger.warning(
                                f"No verdict column found for prediction {i+1}"
                            )
                        results[verdict_col] = 0
                else:
                    # Some dataset.evaluate implementations (e.g., WaltonMultimodalReasoning)
                    # return metrics and write the judged results to a sidecar file
                    try:
                        model_name_for_path = judge_kwargs.get("model", "gpt-4o-mini")
                        suffix = temp_file.split(".")[-1]
                        result_path = temp_file.replace(
                            f".{suffix}", f"_{model_name_for_path}_judge.xlsx"
                        )
                        if osp.exists(result_path):
                            eval_df = load(result_path)
                            if (
                                isinstance(eval_df, pd.DataFrame)
                                and "verdict" in eval_df.columns
                            ):
                                eval_df_for_debug = eval_df
                                results[verdict_col] = eval_df["verdict"].values
                                logger.info(
                                    f"Loaded {len(eval_df)} verdicts for prediction {i+1} from {osp.basename(result_path)}"
                                )
                                loaded_from_file = True
                    except Exception as e:
                        if judge_kwargs.get("verbose", False):
                            logger.warning(
                                f"Failed to load judge sidecar for prediction {i+1}: {e}"
                            )

                    if not loaded_from_file:
                        if judge_kwargs.get("verbose", False):
                            logger.warning(
                                f"Unexpected evaluation result format for prediction {i+1}; defaulting verdicts to 0"
                            )
                        results[verdict_col] = 0

                if isinstance(eval_df_for_debug, pd.DataFrame):
                    if "judge_stage" in eval_df_for_debug.columns:
                        results[judge_stage_col] = eval_df_for_debug["judge_stage"].values
                        temp_df["judge_stage"] = eval_df_for_debug["judge_stage"].values
                        keep_temp_file = True
                    if "judge_stage_detail" in eval_df_for_debug.columns:
                        results[judge_stage_detail_col] = eval_df_for_debug[
                            "judge_stage_detail"
                        ].values
                        temp_df["judge_stage_detail"] = eval_df_for_debug[
                            "judge_stage_detail"
                        ].values
                        keep_temp_file = True
                    if keep_temp_file:
                        temp_df["verdict"] = results[verdict_col].values
                        temp_df.to_excel(temp_file, index=False)

                # Save intermediate results after each prediction evaluation
                intermediate_file = osp.join(
                    work_dir, f"{dataset_name}_evaluated_partial.xlsx"
                )
                results.to_excel(intermediate_file, index=False)
                logger.info(f"Saved checkpoint after evaluating prediction {i+1}")

            except Exception as e:
                logger.error(f"Error evaluating prediction {i+1}: {e}")
                # Save what we have so far
                intermediate_file = osp.join(
                    work_dir, f"{dataset_name}_evaluated_partial.xlsx"
                )
                results.to_excel(intermediate_file, index=False)
                logger.info(f"Saved checkpoint before failing on prediction {i+1}")
                raise

            finally:
                # Keep enriched temp files when they contain per-row judge stage metadata.
                if osp.exists(temp_file) and not keep_temp_file:
                    os.remove(temp_file)

    finally:
        # Clean up judge model if we created it (and it's VLLM)
        if judge_model is not None and hasattr(judge_model, "llm"):
            logger.info("Cleaning up VLLM judge model")
            del judge_model
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate verdict_sum (total correct out of k)
    verdict_columns = [f"verdict_{i+1}" for i in range(k)]
    results["verdict_sum"] = results[verdict_columns].sum(axis=1)

    # Add statistics columns
    results["verdict_mean"] = results["verdict_sum"] / k
    results["difficulty"] = pd.cut(
        results["verdict_mean"],
        bins=[0, 0.25, 0.75, 1.0],
        labels=["hard", "medium", "easy"],
    )

    logger.info(f"Evaluation complete. Verdict distribution:")
    logger.info(f"\n{results['verdict_sum'].value_counts().sort_index()}")

    return results


def merge_kfold_results(work_dir, model_name, dataset_name, k, world_size):
    """Merge k-fold results from multiple ranks into a single file."""
    logger = get_logger("RUN")

    # Load results from all ranks
    all_results = {}
    for rank in range(world_size):
        rank_file = osp.join(
            work_dir, f"{model_name}_{dataset_name}_k{k}_rank{rank}.pkl"
        )
        if osp.exists(rank_file):
            rank_results = load(rank_file)
            all_results.update(rank_results)
            logger.info(f"Loaded {len(rank_results)} results from rank {rank}")

    # Save merged results
    merged_file = osp.join(work_dir, f"{model_name}_{dataset_name}_k{k}.pkl")
    dump(all_results, merged_file)
    logger.info(f"Merged {len(all_results)} total results to {merged_file}")

    # Clean up rank-specific files
    for rank in range(world_size):
        rank_file = osp.join(
            work_dir, f"{model_name}_{dataset_name}_k{k}_rank{rank}.pkl"
        )
        if osp.exists(rank_file):
            os.remove(rank_file)

    return all_results


def main():
    """Main function for k-fold inference."""
    args = parse_args()

    # Initialize logger early
    logger = get_logger("RUN")

    # Initialize distributed if using multiple GPUs
    if WORLD_SIZE > 1:
        import torch.distributed as dist
        from datetime import timedelta

        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=int(os.environ.get("DIST_TIMEOUT", 3600))),
        )

    # Disable warnings if requested
    if args.no_warning:
        warnings.filterwarnings("ignore")

    # Setup work directory
    work_dir = args.work_dir if args.work_dir else "./outputs"
    os.makedirs(work_dir, exist_ok=True)

    # Handle custom model or regular model
    if args.pass_custom_model:
        # Register custom model and get clean name (like run.py does)
        try:
            from vlmeval.utils.model_detection import register_custom_model

            model_name = register_custom_model(args.pass_custom_model)
            logger.info(
                f"Successfully registered custom model: {model_name} -> {args.pass_custom_model}"
            )
            use_custom_model = True
        except Exception as e:
            logger.error(
                f"Failed to register custom model {args.pass_custom_model}: {e}"
            )
            sys.exit(1)
    elif args.model:
        model_name = args.model
        use_custom_model = False
    else:
        logger.error("ERROR: Either --model or --pass-custom-model must be specified")
        sys.exit(1)

    dataset_names = args.data

    # Convert single dataset to list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    logger.info(f"Starting k-fold inference with k={args.k}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Datasets: {dataset_names}")
    logger.info("Using VLLM for acceleration")

    # Build model kwargs
    model_kwargs = {}
    if args.verbose:
        model_kwargs["verbose"] = True
    if args.retry:
        model_kwargs["retry"] = args.retry
    # VLLM is required
    model_kwargs["use_vllm"] = True
    if args.batch_size:
        model_kwargs["batch_size"] = args.batch_size
    if args.max_output_tokens:
        model_kwargs["max_output_tokens"] = args.max_output_tokens
    if args.temperature:
        model_kwargs["temperature"] = args.temperature
    if args.top_p:
        model_kwargs["top_p"] = args.top_p
    # Drive VLLM candidate count from k
    if args.k:
        model_kwargs["n"] = args.k
    if args.force_sequential_multimodal_vllm:
        model_kwargs["force_sequential_on_multimodal_vllm"] = True

    # Handle nproc/api-nproc
    nproc = args.api_nproc if args.api_nproc else args.nproc
    if nproc:
        model_kwargs["nproc"] = nproc

    # Process each dataset
    for dataset_name in dataset_names:
        logger.info(f"\nProcessing dataset: {dataset_name}")

        # Create model-specific work directory (like run.py does)
        model_work_dir = osp.join(work_dir, model_name)
        os.makedirs(model_work_dir, exist_ok=True)

        # Build dataset
        dataset = build_dataset(dataset_name)
        if args.limit is not None:
            original_size = len(dataset.data)
            dataset.data = dataset.data.iloc[: args.limit].copy()
            logger.info(
                f"Debug limit enabled: processing {len(dataset.data)}/{original_size} rows"
            )

        # Check if inference results already exist
        inference_file = osp.join(
            model_work_dir, f"{model_name}_{dataset_name}_k{args.k}.pkl"
        )
        predictions_file = osp.join(
            model_work_dir, f"{model_name}_{dataset_name}_k{args.k}_predictions.xlsx"
        )

        # Only skip inference if existing results are COMPLETE (mirror _infer_kfold_batched)
        if args.reuse and osp.exists(inference_file):
            try:
                existing_results = load(inference_file)
                data = dataset.data
                all_complete = all(
                    existing_results.get(row["index"], {}).get("predictions", None)
                    is not None
                    and len(
                        existing_results.get(row["index"], {}).get("predictions", [])
                    )
                    == args.k
                    for _, row in data.iterrows()
                )
                need_inference = not all_complete
                if not need_inference:
                    logger.info(
                        "All inference results are complete, will reuse without rerun"
                    )
            except Exception:
                # If loading/checking fails, fall back to running inference
                need_inference = True
        else:
            # No reusable file or reuse=False -> run inference
            need_inference = True

        # Evaluation only skipped when the final evaluated file exists and reuse=True
        need_evaluation = args.judge and (
            not osp.exists(
                predictions_file.replace("_predictions.xlsx", "_evaluated.xlsx")
            )
            or not args.reuse
        )

        # Skip everything if both inference and evaluation are already done
        if not need_inference and not need_evaluation:
            logger.info("Both inference and evaluation already completed, skipping...")
            continue

        model = None
        kfold_results = None

        if need_inference:
            logger.info("Building model for inference...")
            # Build the model only if we need to do inference
            if use_custom_model:
                model = build_model(model_name, **model_kwargs)
            else:
                model = build_model(model_name, **model_kwargs)

            # Run k-fold inference (with batch optimization if available)
            kfold_results = infer_kfold_batch(
                model=model,
                dataset=dataset,
                k=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                seed_base=args.seed_base,
                work_dir=model_work_dir,
                verbose=args.verbose,
                reuse=args.reuse,
                batch_size=args.batch_size,
                model_name=model_name,
                skip_image=args.skip_image,
            )
        else:
            logger.info("Inference results exist and reuse=True, skipping inference")
            if need_evaluation:
                # Load existing results for evaluation
                kfold_results = load(inference_file)

        # Synchronize all ranks before merging
        if WORLD_SIZE > 1:
            dist.barrier()

        # Merge results from all ranks (only rank 0 does this)
        if WORLD_SIZE > 1 and RANK == 0:
            kfold_results = merge_kfold_results(
                model_work_dir, model_name, dataset_name, args.k, WORLD_SIZE
            )
        elif WORLD_SIZE > 1:
            # Other ranks wait for merge to complete
            dist.barrier()
            # Load merged results
            merged_file = osp.join(
                model_work_dir, f"{model_name}_{dataset_name}_k{args.k}.pkl"
            )
            kfold_results = load(merged_file)

        # Only rank 0 does evaluation and saving
        if RANK == 0:
            # Convert to DataFrame
            df_predictions = convert_to_dataframe(kfold_results, args.k)

            # Save predictions
            pred_file = osp.join(
                model_work_dir,
                f"{model_name}_{dataset_name}_k{args.k}_predictions.xlsx",
            )
            df_predictions.to_excel(pred_file, index=False)
            logger.info(f"Predictions saved to {pred_file}")

        # Free GPU memory from inference model before evaluation (only if we loaded it)
        if model is not None and hasattr(model, "llm"):
            logger.info("Freeing VLLM GPU memory before evaluation...")
            del model.llm
            del model
            torch.cuda.empty_cache()
            import gc

            gc.collect()
            torch.cuda.synchronize()
            logger.info("GPU memory freed")
            model = None

        # Evaluate if judge is available
        if hasattr(dataset, "evaluate") and RANK == 0 and need_evaluation:
            # Load kfold_results if we skipped inference but need evaluation
            if kfold_results is None:
                logger.info("Loading existing results for evaluation")
                kfold_results = load(inference_file)
                df_predictions = convert_to_dataframe(kfold_results, args.k)

            # Check for partial evaluation results
            partial_eval_file = osp.join(
                model_work_dir, f"{dataset_name}_evaluated_partial.xlsx"
            )
            if osp.exists(partial_eval_file) and args.reuse:
                logger.info(
                    f"Loading partial evaluation results from {partial_eval_file}"
                )
                df_partial = pd.read_excel(partial_eval_file)

                # Check which verdicts are already complete
                completed_verdicts = []
                for i in range(args.k):
                    verdict_col = f"verdict_{i+1}"
                    if (
                        verdict_col in df_partial.columns
                        and not df_partial[verdict_col].isna().all()
                    ):
                        completed_verdicts.append(i + 1)

                if completed_verdicts:
                    logger.info(
                        f"Found partial verdicts for predictions {completed_verdicts}"
                    )
                    # Use partial results as starting point
                    df_predictions = df_partial
                else:
                    logger.info(
                        "Partial file exists but contains no verdicts, starting fresh"
                    )

            # Determine if we should use VLLM for judge
            use_vllm_judge = args.judge and not args.judge.startswith("gpt")

            logger.info(f"Starting evaluation with judge: {args.judge}")
            judge_kwargs = {
                "model": args.judge if args.judge else "gpt-4o-mini",
                "batch_size": (
                    args.judge_batch_size if args.judge_batch_size else 32
                ),
                "use_vllm_judge": use_vllm_judge,
                "nproc": args.api_nproc if args.api_nproc else args.nproc,
                "verbose": args.verbose,
                "walton_judge_impl": args.walton_judge_impl,
            }

            df_evaluated = evaluate_kfold(
                dataset=dataset,
                df_predictions=df_predictions,
                k=args.k,
                work_dir=model_work_dir,  # Use model-specific directory
                **judge_kwargs,
            )

            # Save evaluated results
            eval_file = osp.join(
                model_work_dir, f"{model_name}_{dataset_name}_k{args.k}_evaluated.xlsx"
            )
            # Do not persist auxiliary columns in the saved file
            df_to_save = df_evaluated.drop(
                columns=["verdict_mean", "difficulty"], errors="ignore"
            )
            df_to_save.to_excel(eval_file, index=False)
            logger.info(f"Evaluated results saved to {eval_file}")

            # Print summary statistics
            logger.info("\n" + "=" * 60)
            logger.info("K-FOLD EVALUATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total questions: {len(df_evaluated)}")
            logger.info(f"K value: {args.k}")
            logger.info("\nDifficulty distribution:")
            logger.info(df_evaluated["difficulty"].value_counts())
            logger.info("\nVerdict sum distribution:")
            for i in range(args.k + 1):
                count = (df_evaluated["verdict_sum"] == i).sum()
                pct = count / len(df_evaluated) * 100
                logger.info(f"  {i}/{args.k} correct: {count} ({pct:.1f}%)")

    logger.info("\n✅ K-fold inference and evaluation complete!")


if __name__ == "__main__":
    main()
