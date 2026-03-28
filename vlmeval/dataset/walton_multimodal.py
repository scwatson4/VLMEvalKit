import os.path as osp
import io
import json
import re
import pandas as pd
from PIL import Image
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich


class WaltonMultimodalReasoning(ImageBaseDataset):
    TYPE = "VQA"

    DATASET_URL = {
        "WaltonMultimodalColdStart": "oumi-ai/walton-multimodal-cold-start-r1-format",
        "MM_MathInstruct": "oumi-ai/MM-MathInstruct-to-r1-format-filtered",
        "MultimodalOpenR1_8192_Filtered_Mid_IC": "oumi-ai/multimodal-open-r1-8192-filtered-mid-ic",
        "MultimodalOpenR1_8K_Verified": "lmms-lab/multimodal-open-r1-8k-verified",
    }
    DATASET_MD5 = {}

    def _build_vllm_judge(self, model_name, batch_size=32, **kwargs):
        """Build a VLLM-based judge model for efficient batch evaluation."""
        try:
            from vllm import LLM, SamplingParams
            import torch

            # Map model names to actual paths
            model_path = model_name
            if model_name == "qwen3-4b":
                model_path = "Qwen/Qwen3-4B-Instruct-2507"
            elif model_name == "llama31-8b":
                model_path = "meta-llama/Llama-3.1-8B-Instruct"

            # Initialize VLLM with appropriate settings
            gpu_count = torch.cuda.device_count()
            tp_size = min(gpu_count, 4)  # Use at most 4 GPUs for judge model

            vllm_params = {
                "model": model_path,
                "max_num_seqs": batch_size,
                "tensor_parallel_size": tp_size,
                "gpu_memory_utilization": 0.9,  # Make sure to unload main model before evaluation
                "max_model_len": 16384,  # Judge prompts are short
            }

            from ..smp import get_logger

            logger = get_logger("VLLM_JUDGE")

            logger.info("=== VLLM Judge Instantiation Parameters ===")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Batch size (max_num_seqs): {batch_size}")
            logger.info(f"Tensor parallel size: {tp_size}")
            logger.info(f"GPU count available: {gpu_count}")
            logger.info(
                f"GPU memory utilization: {vllm_params['gpu_memory_utilization']}"
            )
            logger.info(f"Max model length: {vllm_params['max_model_len']}")
            logger.info(f"Additional kwargs: {kwargs}")
            logger.info("=" * 50)

            llm = LLM(**vllm_params)

            # Wrap in a simple interface
            class VLLMJudge:
                def __init__(self, llm_instance, max_model_len):
                    self.llm = llm_instance
                    self.max_model_len = max_model_len
                    self.sampling_params = SamplingParams(
                        temperature=0.1,
                        max_tokens=256,
                        # Remove aggressive stop tokens that can cause empty outputs
                        stop=None,
                    )

                def _get_tokenizer(self):
                    try:
                        return self.llm.get_tokenizer()
                    except Exception:
                        return None

                def _log_overlong_prompts(self, prompts, logger):
                    tokenizer = self._get_tokenizer()
                    for idx, prompt in enumerate(prompts):
                        prompt_len = None
                        if tokenizer is not None:
                            try:
                                prompt_len = len(tokenizer.encode(prompt))
                            except Exception:
                                prompt_len = None

                        if prompt_len is None or prompt_len > self.max_model_len:
                            preview = " ".join(str(prompt).split())[:400]
                            logger.error(
                                "Judge prompt %d may exceed max_model_len=%d (token_len=%s). Preview: %s",
                                idx,
                                self.max_model_len,
                                prompt_len if prompt_len is not None else "unknown",
                                preview,
                            )

                def generate(self, prompts):
                    """Generate responses for batch of prompts."""
                    if isinstance(prompts, str):
                        prompts = [prompts]

                    from ..smp import get_logger

                    logger = get_logger("VLLM_JUDGE")

                    logger.info("=== VLLM Judge Generation ===")
                    logger.info(f"Processing batch of {len(prompts)} prompts")
                    logger.info(
                        f"Sampling params: temp={self.sampling_params.temperature}, max_tokens={self.sampling_params.max_tokens}"
                    )
                    logger.info("=" * 30)

                    try:
                        outputs = self.llm.generate(prompts, self.sampling_params)
                    except ValueError as err:
                        if "maximum model length" in str(err):
                            logger.error(
                                "Encountered an overlong Walton judge prompt. Logging candidate prompts before re-raising."
                            )
                            self._log_overlong_prompts(prompts, logger)
                        raise
                    responses = [output.outputs[0].text for output in outputs]

                    logger.info(f"Generated {len(responses)} responses")
                    return responses if len(responses) > 1 else responses[0]

                def __del__(self):
                    """Clean up VLLM resources."""
                    if hasattr(self, "llm"):
                        del self.llm
                        torch.cuda.empty_cache()

            return VLLMJudge(llm, vllm_params["max_model_len"])

        except ImportError:
            # Fallback to regular judge if VLLM not available
            return build_judge(**kwargs)

    def __init__(self, dataset="WaltonMultimodalReasoning", **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset_name = dataset

    def _extract_answer_text(self, text):
        """Extract a likely final answer string while preserving raw text elsewhere."""
        if not isinstance(text, str):
            return ""

        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()

        tag_patterns = [
            r"<ans>(.*?)</ans>",
            r"<answer>(.*?)</answer>",
            r"<final>(.*?)</final>",
            r"\[ANSWER\](.*?)\[/ANSWER\]",
        ]
        for pattern in tag_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        return text.strip()

    def _extract_options_block(self, question_text):
        if not isinstance(question_text, str):
            return None
        match = re.search(
            r"\n(?:Choices|Options)\s*:?[\t ]*\n",
            question_text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        tail = question_text[match.end() :]
        block_lines = []
        for raw in tail.splitlines():
            if raw.strip() == "":
                break
            block_lines.append(raw)
        if not block_lines:
            return None
        return "\n".join(block_lines)

    def _build_choice_map_for_qtext(self, question_text):
        options_block = self._extract_options_block(question_text)
        if not options_block:
            return {}

        lines = options_block.splitlines()
        labeled_mapping = {}
        labeled_found = False
        for raw in lines:
            match = re.match(r"^\s*(?:[-*•]\s*)?([A-Za-z])[\.)]\s*(.+)$", raw)
            if match:
                labeled_found = True
                labeled_mapping[match.group(1).upper()] = match.group(2).strip()
        if labeled_found and labeled_mapping:
            return labeled_mapping

        unlabeled_mapping = {}
        base = ord("A")
        for i, raw in enumerate(lines):
            text = re.sub(r"^\s*[-*•]\s*", "", raw.strip())
            unlabeled_mapping[chr(base + i)] = text
        return unlabeled_mapping

    def _normalize_answer_with_choices(self, answer, choice_map):
        if not isinstance(answer, str):
            return answer, None

        stripped = answer.strip()
        match = re.match(r"^\s*([A-Za-z])[\.)]\s*(.+)$", stripped)
        if match:
            label = match.group(1).upper()
            rest = match.group(2).strip()
            mapped = choice_map.get(label, rest) if isinstance(choice_map, dict) else rest
            return label, mapped
        if len(stripped) == 1 and stripped.upper() in choice_map:
            return stripped.upper(), choice_map[stripped.upper()]
        if len(stripped) == 2 and stripped[1] in ".)" and stripped[0].upper() in choice_map:
            label = stripped[0].upper()
            return label, choice_map[label]
        return stripped, None

    def _texts_match_basic(self, left, right):
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        left_norm = re.sub(r"\s+", " ", left).strip().lower()
        right_norm = re.sub(r"\s+", " ", right).strip().lower()
        return bool(left_norm) and left_norm == right_norm

    def _deterministic_verdict(self, prediction, answer, question_text=None):
        """Return (verdict, stage_name, detail) if deterministic normalization is sufficient."""
        raw_prediction = str(prediction)
        raw_answer = str(answer)

        if self._texts_match_basic(raw_prediction, raw_answer):
            return (1, "deterministic_raw_match", "Normalized raw prediction matches raw answer")

        choice_map = self._build_choice_map_for_qtext(question_text)
        extracted_prediction = self._extract_answer_text(raw_prediction)
        extracted_answer = self._extract_answer_text(raw_answer)

        if self._texts_match_basic(extracted_prediction, extracted_answer):
            return (
                1,
                "deterministic_extracted_match",
                f"Extracted answers match: pred={extracted_prediction!r}, answer={extracted_answer!r}",
            )

        pred_label, pred_text = self._normalize_answer_with_choices(
            extracted_prediction, choice_map
        )
        ans_label, ans_text = self._normalize_answer_with_choices(
            extracted_answer, choice_map
        )

        if (
            isinstance(pred_label, str)
            and isinstance(ans_label, str)
            and len(pred_label) == 1
            and len(ans_label) == 1
            and pred_label.upper() == ans_label.upper()
        ):
            return (
                1,
                "deterministic_choice_label",
                f"Choice labels match: pred={pred_label}, answer={ans_label}",
            )

        if pred_text is not None and ans_text is not None and self._texts_match_basic(
            pred_text, ans_text
        ):
            return (
                1,
                "deterministic_choice_text",
                f"Choice texts match: pred={pred_text!r}, answer={ans_text!r}",
            )

        if self._texts_match_basic(extracted_prediction, raw_answer):
            return (
                1,
                "deterministic_extracted_to_raw",
                f"Extracted prediction matches raw answer: pred={extracted_prediction!r}, answer={raw_answer!r}",
            )

        return None

    def _create_multistage_judge_prompt(self, prediction, ground_truth, question_text=None):
        options_block = self._extract_options_block(question_text)

        prompt = """You are evaluating whether a model prediction should be counted as correct for a reasoning task.

You must compare the model prediction against the ground truth answer for semantic or mathematical equivalence.
The model prediction may include long reasoning traces and final answers wrapped in formats such as \\boxed{}, <ans></ans>, or similar tags. Extract the final answer internally if needed, but judge from the full raw prediction and full raw ground truth provided below.

Question:
"""
        prompt += f"{question_text if isinstance(question_text, str) else ''}\n\n"
        prompt += f"""Model Prediction (raw):
{prediction}

Ground Truth Answer (raw):
{ground_truth}
"""

        if options_block:
            prompt += f"\nMultiple Choice Options:\n{options_block}\n"

        prompt += """
Count answers as equivalent when they match in semantic meaning, mathematical value, units where appropriate, or correct multiple-choice label/content.

Return EXACTLY one JSON object and nothing else:
{"equivalent": true, "reasoning": "brief explanation"}
or
{"equivalent": false, "reasoning": "brief explanation"}
"""
        return prompt

    def _parse_multistage_judge_response(self, response):
        if not isinstance(response, str):
            return 0, "Non-string judge response"

        text = response.strip()
        try:
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0]
            parsed = json.loads(text)
            equivalent = parsed.get("equivalent", False)
            reasoning = parsed.get("reasoning", "")
            return (1 if equivalent else 0), f"json:{reasoning}"
        except Exception:
            pass

        match = re.search(r'\{[^}]*"equivalent"\s*:\s*(true|false)', response, flags=re.IGNORECASE)
        if match:
            verdict = 1 if match.group(1).lower() == "true" else 0
            return verdict, "regex_json_fallback"

        lowered = text.lower()
        if re.search(r"\bfalse\b", lowered) or re.search(r"\bnot equivalent\b", lowered):
            return 0, "phrase_fallback_false"
        if re.search(r"\btrue\b", lowered) or re.search(r"\bequivalent\b", lowered):
            return 1, "phrase_fallback_true"
        return 0, "unparsed_response"

    def _run_judge_generation_batch(
        self,
        judge_model,
        prompts,
        use_vllm_judge=False,
        judge_nproc=4,
    ):
        if not prompts:
            return []

        if use_vllm_judge:
            responses = judge_model.generate(prompts)
            if not isinstance(responses, list):
                responses = [responses]
            return responses

        if len(prompts) == 1:
            return [judge_model.generate(prompts[0])]

        if getattr(judge_model, "is_api", False) and judge_nproc > 1:
            return track_progress_rich(
                judge_model.generate,
                [(prompt,) for prompt in prompts],
                nproc=judge_nproc,
            )

        return [judge_model.generate(prompt) for prompt in prompts]

    def _evaluate_multistage_judge(self, eval_file, judge_model=None, **judge_kwargs):
        model = judge_kwargs.get("model", "gpt-4o-mini")
        suffix = eval_file.split(".")[-1]
        result_path = eval_file.replace(f".{suffix}", f"_{model}_judge.xlsx")
        score_path = eval_file.replace(f".{suffix}", f"_{model}_score.csv")
        batch_size = judge_kwargs.pop("batch_size", 32)
        judge_nproc = max(1, int(judge_kwargs.get("nproc", 4)))

        if not osp.exists(result_path):
            data = load(eval_file)

            if judge_model is None:
                use_vllm_judge = judge_kwargs.get("use_vllm_judge", False)
                judge_kwargs["model"] = model
                if use_vllm_judge and not model.startswith("gpt"):
                    judge_model = self._build_vllm_judge(
                        model, batch_size=batch_size, **judge_kwargs
                    )
                else:
                    judge_model = build_judge(**judge_kwargs)
            else:
                use_vllm_judge = hasattr(judge_model, "llm")

            if hasattr(judge_model, "working"):
                assert judge_model.working(), (
                    "WaltonMultimodalReasoning evaluation requires a working judge model\n"
                    + DEBUG_MESSAGE
                )

            logger = get_logger("WALTON_EVAL")
            verdict_list = [0] * len(data)
            judge_stage_list = [""] * len(data)
            judge_stage_detail_list = [""] * len(data)
            logged_sample = False

            pending_requests = []
            pending_meta = []

            def flush_pending_requests():
                nonlocal logged_sample, pending_requests, pending_meta
                if not pending_requests:
                    return

                try:
                    batch_responses = self._run_judge_generation_batch(
                        judge_model=judge_model,
                        prompts=pending_requests,
                        use_vllm_judge=use_vllm_judge,
                        judge_nproc=judge_nproc,
                    )
                except Exception as e:
                    print(f"Error in multistage judge generation: {e}")
                    batch_responses = [""] * len(pending_requests)

                for meta, response in zip(pending_meta, batch_responses):
                    verdict, detail = self._parse_multistage_judge_response(response)
                    verdict_list[meta["abs_idx"]] = verdict
                    judge_stage_list[meta["abs_idx"]] = "llm_multistage"
                    judge_stage_detail_list[meta["abs_idx"]] = detail

                    deterministic = self._deterministic_verdict(
                        meta["prediction"],
                        meta["answer"],
                        meta["question"],
                    )
                    if deterministic is not None and deterministic[0] == 1 and verdict == 0:
                        verdict_list[meta["abs_idx"]] = 1
                        judge_stage_list[meta["abs_idx"]] = "llm_multistage_corrected_by_deterministic"
                        judge_stage_detail_list[meta["abs_idx"]] = deterministic[2]

                if not logged_sample and pending_requests:
                    try:
                        logger.info("=== First multistage Walton judge batch ===")
                        for meta, prompt, response in zip(
                            pending_meta[:3], pending_requests[:3], batch_responses[:3]
                        ):
                            logger.info(
                                f"--- Judged Item (abs {meta['abs_idx']}) ---\nPrompt:\n{prompt}\n\nResponse:\n{str(response)}"
                            )
                    except Exception:
                        pass
                    logged_sample = True

                pending_requests = []
                pending_meta = []

            with tqdm(total=len(data), desc="Evaluating with multistage judge") as pbar:
                for abs_idx, (_, row) in enumerate(data.iterrows()):
                    prediction = row.get("prediction", "")
                    answer = row.get("answer", "")
                    question = row.get("question", None)
                    deterministic = self._deterministic_verdict(
                        prediction,
                        answer,
                        question,
                    )

                    if deterministic is not None:
                        verdict_list[abs_idx] = deterministic[0]
                        judge_stage_list[abs_idx] = deterministic[1]
                        judge_stage_detail_list[abs_idx] = deterministic[2]
                    else:
                        pending_requests.append(
                            self._create_multistage_judge_prompt(
                                prediction,
                                answer,
                                question,
                            )
                        )
                        pending_meta.append(
                            {
                                "abs_idx": abs_idx,
                                "prediction": prediction,
                                "answer": answer,
                                "question": question,
                            }
                        )
                        if len(pending_requests) >= batch_size:
                            flush_pending_requests()

                    pbar.update(1)

                flush_pending_requests()

            data["verdict"] = verdict_list
            data["judge_stage"] = judge_stage_list
            data["judge_stage_detail"] = judge_stage_detail_list
            dump(data, result_path)

        data = load(result_path)
        overall_acc = data["verdict"].mean() * 100
        score_df = pd.DataFrame({"Metric": ["Overall Accuracy"], "Value": [overall_acc]})
        dump(score_df, score_path)

        ret = {"Overall": overall_acc}
        if "category" in data.columns:
            categories = data["category"].unique()
            for cat in categories:
                cat_data = data[data["category"] == cat]
                ret[cat] = cat_data["verdict"].mean() * 100
        return ret

    def _normalize_image_field(self, image_value):
        """Normalize Hugging Face image payloads into base64 strings for VLMEvalKit."""
        if image_value is None:
            return ""

        # Standard HF Image feature representation: {"bytes": ..., "path": ...}
        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            image_path = image_value.get("path")

            if image_bytes:
                if isinstance(image_bytes, (bytes, bytearray)):
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        return encode_image_to_base64(img)
                if isinstance(image_bytes, str):
                    # Some datasets already provide base64 or another string serialization.
                    return image_bytes

            if image_path and osp.exists(image_path):
                return encode_image_file_to_base64(image_path)

            return ""

        # PIL image-like object from datasets.Image
        if hasattr(image_value, "save") and hasattr(image_value, "mode"):
            return encode_image_to_base64(image_value)

        # Already serialized as a string (base64, URL, or path-like)
        if isinstance(image_value, str):
            if osp.exists(image_value):
                return encode_image_file_to_base64(image_value)
            return image_value

        return ""

    def prepare_dataset(self, dataset):
        # Load dataset from HuggingFace
        ROOT = LMUDataRoot()
        data_file = osp.join(ROOT, f"{dataset}.tsv")

        if not osp.exists(data_file):
            repo_id = self.DATASET_URL.get(dataset)
            if not repo_id:
                raise ValueError(f"Unsupported dataset: {dataset}")

            # Import datasets library only when needed
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "The 'datasets' library is required to load WaltonMultimodalReasoning dataset. "
                    "Please install it with: pip install datasets"
                )

            # Import encode function here to avoid circular import
            from ..tools import encode_image_to_base64

            # Load from HuggingFace
            hf_dataset = load_dataset(repo_id, split="train")

            # Convert to tsv format expected by VLMEvalKit
            data_list = []
            for idx, item in enumerate(hf_dataset):
                # The problem field contains both image reference and question
                problem_text = item["problem"]

                # Normalize image payloads into base64 for downstream dump_image().
                image_data = self._normalize_image_field(item.get("image"))

                data_list.append(
                    {
                        "index": idx,
                        "image": image_data,
                        "question": problem_text,
                        "answer": item["solution"],
                    }
                )

            # Save as TSV
            df = pd.DataFrame(data_list)
            df.to_csv(data_file, sep="\t", index=False)

        return data_file

    def load_data(self, dataset):
        data_file = self.prepare_dataset(dataset)
        return load(data_file)

    def build_prompt(self, line):
        # Build the prompt with the reasoning trace structure
        prompt = """Put your final answer within \\boxed{}.

"""

        if isinstance(line, int):
            line = self.data.iloc[line]

        # Add the question
        question = line.get("question", "")
        prompt += f"Question: {question}"

        msgs = [{"type": "text", "value": prompt}]

        # Add image if present
        if "image" in line and line["image"]:
            # Handle image path or base64 encoding
            image_paths = self.dump_image(line)
            # dump_image returns a list, take the first element for single image
            if image_paths:
                msgs.append({"type": "image", "value": image_paths[0]})

        return msgs

    def evaluate(self, eval_file, judge_model=None, **judge_kwargs):
        walton_judge_impl = judge_kwargs.get("walton_judge_impl", "default")
        if walton_judge_impl == "multistage":
            return self._evaluate_multistage_judge(
                eval_file, judge_model=judge_model, **judge_kwargs
            )

        # Use GPT-4o-mini as judge for evaluation
        model = judge_kwargs.get("model", "gpt-4o-mini")
        suffix = eval_file.split(".")[-1]
        result_path = eval_file.replace(f".{suffix}", f"_{model}_judge.xlsx")
        score_path = eval_file.replace(f".{suffix}", f"_{model}_score.csv")
        batch_size = judge_kwargs.pop(
            "batch_size", 32
        )  # Use proper batch_size parameter
        judge_nproc = max(1, int(judge_kwargs.get("nproc", 4)))

        if not osp.exists(result_path):
            data = load(eval_file)

            # Use provided judge model or build a new one
            if judge_model is None:
                # Check if we should use VLLM for judge (for local models like qwen3-4b)
                use_vllm_judge = judge_kwargs.get("use_vllm_judge", False)

                # Build judge model
                judge_kwargs["model"] = model
                if use_vllm_judge and not model.startswith("gpt"):
                    # Use VLLM for local judge models
                    judge_model = self._build_vllm_judge(
                        model, batch_size=batch_size, **judge_kwargs
                    )
                else:
                    judge_model = build_judge(**judge_kwargs)
            else:
                # Using pre-built judge model
                use_vllm_judge = hasattr(
                    judge_model, "llm"
                )  # Check if it's a VLLM model

            # Check if judge is working (only for API models)
            if hasattr(judge_model, "working"):
                assert judge_model.working(), (
                    "WaltonMultimodalReasoning evaluation requires a working judge model\n"
                    + DEBUG_MESSAGE
                )

            def extract_answer(text):
                """Extract the answer from \\boxed{} format"""
                import re

                pattern = r"\\boxed\{([^}]*)\}"
                matches = re.findall(pattern, text)
                if matches:
                    return matches[-1].strip()
                return text.strip()

            def create_judge_prompt(prediction, ground_truth, question_text=None):
                """Create a judge prompt for a single prediction with optional choices mapping."""
                import re

                def extract_options_block(qtext):
                    if not qtext:
                        return None
                    m = re.search(
                        r"\n(?:Choices|Options)\s*:?[\t ]*\n",
                        qtext,
                        flags=re.IGNORECASE,
                    )
                    if not m:
                        return None
                    tail = qtext[m.end() :]
                    block_lines = []
                    for raw in tail.splitlines():
                        if raw.strip() == "":
                            break
                        # Preserve each line exactly as-is
                        block_lines.append(raw)
                    if not block_lines:
                        return None
                    return "\n".join(block_lines)

                pred_answer = extract_answer(str(prediction))
                gt_answer = extract_answer(str(ground_truth))
                choices_map = (
                    extract_options_block(question_text) if question_text else None
                )

                prompt = f"""You are evaluating a model's answer against the ground truth for a reasoning problem.

Model's Answer: {pred_answer}

Ground Truth: {gt_answer}
"""
                if choices_map:
                    prompt += f"\nMultiple Choice Options:\n{choices_map}\n"

                prompt += """
Please evaluate whether the model's answer is correct compared to the ground truth. Consider:
1. Mathematical equivalence (e.g., 58% and 58 are the same)
2. Numerical precision (allow for minor rounding differences)
3. Unit consistency (if units are provided)
"""

                if choices_map:
                    prompt += """
4. Option label/content equivalence when choices are provided (e.g., "A. 48°" and "A" are the same)
"""

                # Deterministic normalization rules to reduce judge ambiguity
                prompt += """
Normalization rules:
 - Trim whitespace and compare case-insensitively for both answers.
 - If both answers are single-letter option labels (A/B/C/D), compare labels directly.
 - If one is a label and the other is text, map the label to its option text and compare texts.
 - If options are provided without labels (one per line), treat them as A, B, C... in order.
 - If normalized Model's Answer and Ground Truth are identical, return {"verdict": 1}.
"""

                prompt += """

Respond with EXACTLY ONE of the following JSON objects and NOTHING ELSE (no code fences, no explanations):
{"verdict": 1}
{"verdict": 0}

Where the value of "verdict" must be the integer 1 if the model's answer is correct, or the integer 0 if it is incorrect.
"""
                return prompt

            def parse_judge_response(response):
                """Parse the judge's response to extract verdict robustly.

                Strategy:
                1) Try strict JSON parse (after stripping ```json fences).
                2) Regex extract a verdict number anywhere in the text.
                3) Accept bare '1'/'0'.
                4) Phrase-based fallback: 'incorrect' => 0; 'correct' => 1.
                Default to 0.
                """
                import re
                import json

                if not isinstance(response, str):
                    return 0

                text = response.strip()

                # 1) Strip code fences and try JSON
                try:
                    if "```json" in text:
                        text = text.split("```json", 1)[1].split("```", 1)[0]
                    elif "```" in text:
                        # Generic fence – keep inner but JSON may still parse
                        text = text.split("```", 1)[1].split("```", 1)[0]
                    result = json.loads(text)
                    verdict = result.get("verdict", 0)
                    verdict = int(verdict)
                    return 1 if verdict >= 1 else 0
                except Exception:
                    pass

                # 2) Regex extract {"verdict": <num>} anywhere
                m = re.search(r'\{[^}]*"verdict"\s*:\s*([-+]?\d+(?:\.\d+)?)', response)
                if m:
                    try:
                        val = float(m.group(1))
                        return 1 if val >= 1 else 0
                    except Exception:
                        pass

                # 3) Bare integer string
                bare = text.strip().lower()
                if bare in ["1", "0"]:
                    return int(bare)

                # 4) Phrase-based fallback – check 'incorrect' first to avoid substring trap
                if re.search(r"\bincorrect\b", bare):
                    return 0
                if re.search(r"\bcorrect\b", bare):
                    return 1

                return 0

            # Helper utilities for option parsing and answer normalization (used pre- and post-judge)
            import re as _re

            def _build_choice_map_for_qtext(qtext):
                if not isinstance(qtext, str):
                    return {}
                m = _re.search(
                    r"\n(?:Choices|Options)\s*:?[\t ]*\n", qtext, flags=_re.IGNORECASE
                )
                if not m:
                    return {}
                tail = qtext[m.end() :]
                lines = []
                for raw in tail.splitlines():
                    if raw.strip() == "":
                        break
                    lines.append(raw)
                # Labeled first
                labeled_mapping = {}
                labeled_found = False
                for raw in lines:
                    mm = _re.match(r"^\s*(?:[-*•]\s*)?([A-Za-z])[\.)]\s*(.+)$", raw)
                    if mm:
                        labeled_found = True
                        labeled_mapping[mm.group(1).upper()] = mm.group(2).strip()
                if labeled_found and len(labeled_mapping):
                    return labeled_mapping
                # Unlabeled fallback A, B, C...
                unlabeled_mapping = {}
                base = ord("A")
                for i, raw in enumerate(lines):
                    text = raw.strip()
                    text = _re.sub(r"^\s*[-*•]\s*", "", text)
                    unlabeled_mapping[chr(base + i)] = text
                return unlabeled_mapping

            def _normalize_answer_with_choices(ans, cmap):
                if not isinstance(ans, str):
                    return ans, None
                s = ans.strip()
                # Handle "A. some string" or "A) some string" by extracting the label and text
                m = _re.match(r"^\s*([A-Za-z])[\.)]\s*(.+)$", s)
                if m:
                    label = m.group(1).upper()
                    rest = m.group(2).strip()
                    mapped = cmap.get(label, rest) if isinstance(cmap, dict) else rest
                    return label, mapped
                if len(s) == 1 and s.upper() in cmap:
                    return s.upper(), cmap[s.upper()]
                if len(s) == 2 and s[1] in ".)" and s[0].upper() in cmap:
                    return s[0].upper(), cmap[s[0].upper()]
                return s, None

            # Process in batches
            verdict_list = []
            total_items = len(data)

            from tqdm import tqdm

            # Debug logging of first prompt/response
            logger = get_logger("WALTON_EVAL")
            logged_sample = False

            with tqdm(total=total_items, desc="Evaluating with judge model") as pbar:
                for i in range(0, total_items, batch_size):
                    # Get batch of data
                    batch_end = min(i + batch_size, total_items)
                    batch_data = data.iloc[i:batch_end]

                    # Pre-judge shortcut: direct normalized match => verdict=1, skip LLM judge
                    batch_prompts = []
                    judge_indices = (
                        []
                    )  # indices (relative to batch_data) that need LLM judging
                    pre_verdicts = [None] * len(batch_data)

                    # Build prompts only for those needing judge
                    for rel_idx, (_, row) in enumerate(batch_data.iterrows()):
                        qtext = row.get("question", None)
                        cmap = _build_choice_map_for_qtext(qtext)
                        pred_ex = extract_answer(str(row.get("prediction", "")))
                        gt_ex = extract_answer(str(row.get("answer", "")))
                        p_label, p_text = _normalize_answer_with_choices(pred_ex, cmap)
                        g_label, g_text = _normalize_answer_with_choices(gt_ex, cmap)

                        equal = False
                        if (
                            isinstance(p_label, str)
                            and isinstance(g_label, str)
                            and len(p_label) == 1
                            and len(g_label) == 1
                        ):
                            equal = p_label.upper() == g_label.upper()
                        if not equal and p_text is not None and g_text is not None:
                            equal = p_text.strip().lower() == g_text.strip().lower()
                        if (
                            not equal
                            and isinstance(pred_ex, str)
                            and isinstance(gt_ex, str)
                        ):
                            equal = pred_ex.strip().lower() == gt_ex.strip().lower()

                        if equal:
                            pre_verdicts[rel_idx] = 1
                        else:
                            judge_indices.append(rel_idx)
                            batch_prompts.append(
                                create_judge_prompt(
                                    row["prediction"], row["answer"], qtext
                                )
                            )

                    # Log the entire first batch with colocated prompt/response pairs

                    # Generate responses for the batch
                    if use_vllm_judge:
                        # VLLM supports true batch processing
                        batch_responses = judge_model.generate(batch_prompts)
                        if not isinstance(batch_responses, list):
                            batch_responses = [batch_responses]
                    else:
                        # For HFChatModel or API models, use sequential generation
                        try:
                            if len(batch_prompts) == 1:
                                batch_responses = [
                                    judge_model.generate(batch_prompts[0])
                                ]
                            elif getattr(judge_model, "is_api", False) and judge_nproc > 1:
                                batch_responses = track_progress_rich(
                                    judge_model.generate,
                                    [(prompt,) for prompt in batch_prompts],
                                    nproc=judge_nproc,
                                )
                            else:
                                # Local non-vLLM judges are kept sequential to avoid
                                # thread-safety and device contention issues.
                                batch_responses = [
                                    judge_model.generate(prompt)
                                    for prompt in batch_prompts
                                ]
                        except Exception as e:
                            print(f"Error in batch generation: {e}")
                            batch_responses = [""] * len(batch_prompts)

                    if not logged_sample and (
                        len(batch_responses) > 0
                        or any(v == 1 for v in pre_verdicts if v is not None)
                    ):
                        try:
                            logger.info(
                                "=== First batch judge prompt/response pairs ==="
                            )
                            # Log pre-judged items
                            for rel_idx, v in enumerate(pre_verdicts):
                                if v == 1:
                                    logger.info(
                                        f"--- Item {rel_idx} ---\nSkipped LLM judge due to direct normalized match => verdict: 1"
                                    )
                            # Log judged items
                            for j, (p, r) in enumerate(
                                zip(batch_prompts, batch_responses)
                            ):
                                logger.info(
                                    f"--- Judged Item (rel {judge_indices[j]}) ---\nPrompt:\n{p}\n\nResponse:\n{str(r)}"
                                )
                        except Exception:
                            pass
                        logged_sample = True

                    # Parse responses
                    # Parse responses for judged subset
                    judged_verdicts = [
                        parse_judge_response(resp) for resp in batch_responses
                    ]

                    # Merge pre-judged and judged verdicts into original batch order
                    batch_verdicts = [0] * len(batch_data)
                    # Fill pre-judged = 1
                    for rel_idx, v in enumerate(pre_verdicts):
                        if v is not None:
                            batch_verdicts[rel_idx] = v
                    # Fill judged results
                    for k, rel_idx in enumerate(judge_indices):
                        batch_verdicts[rel_idx] = judged_verdicts[k]

                    # Post-judge safeguard: if normalized answers match, force verdict=1
                    try:
                        import re

                        def build_choice_map(qtext):
                            if not isinstance(qtext, str):
                                return {}
                            # Find the start of choices/options block
                            m = re.search(
                                r"\n(?:Choices|Options)\s*:?[\t ]*\n",
                                qtext,
                                flags=re.IGNORECASE,
                            )
                            if not m:
                                return {}
                            tail = qtext[m.end() :]
                            lines = []
                            for raw in tail.splitlines():
                                if raw.strip() == "":
                                    break
                                lines.append(raw)

                            # Try labeled pattern first (allow bullets before labels)
                            labeled_mapping = {}
                            labeled_found = False
                            for raw in lines:
                                mm = re.match(
                                    r"^\s*(?:[-*•]\s*)?([A-Za-z])[\.)]\s*(.+)$", raw
                                )
                                if mm:
                                    labeled_found = True
                                    labeled_mapping[mm.group(1).upper()] = mm.group(
                                        2
                                    ).strip()

                            if labeled_found and len(labeled_mapping):
                                return labeled_mapping

                            # Fallback: unlabeled lines – assign A, B, C ... in order
                            unlabeled_mapping = {}
                            base = ord("A")
                            for i, raw in enumerate(lines):
                                text = raw.strip()
                                # Remove common bullet markers
                                text = re.sub(r"^\s*[-*•]\s*", "", text)
                                unlabeled_mapping[chr(base + i)] = text
                            return unlabeled_mapping

                        def normalize_answer(ans, cmap):
                            if not isinstance(ans, str):
                                return ans, None
                            s = ans.strip()
                            # Handle "A. some string" or "A) some string"
                            mm = re.match(r"^\s*([A-Za-z])[\.)]\s*(.+)$", s)
                            if mm:
                                label = mm.group(1).upper()
                                rest = mm.group(2).strip()
                                mapped = (
                                    cmap.get(label, rest)
                                    if isinstance(cmap, dict)
                                    else rest
                                )
                                return label, mapped
                            # If single letter option
                            if len(s) == 1 and s.upper() in cmap:
                                return s.upper(), cmap[s.upper()]
                            # If label with trailing dot like 'A.'
                            if len(s) == 2 and s[1] in ".[)" and s[0].upper() in cmap:
                                return s[0].upper(), cmap[s[0].upper()]
                            return s, None

                        corrected = False
                        for j, ((idx, row), resp) in enumerate(
                            zip(batch_data.iterrows(), batch_responses)
                        ):
                            cmap = build_choice_map(row.get("question", None))
                            pred_ex = extract_answer(str(row.get("prediction", "")))
                            gt_ex = extract_answer(str(row.get("answer", "")))
                            p_label, p_text = normalize_answer(pred_ex, cmap)
                            g_label, g_text = normalize_answer(gt_ex, cmap)

                            equal = False
                            # Compare by label if both labels
                            if (
                                isinstance(p_label, str)
                                and isinstance(g_label, str)
                                and len(p_label) == 1
                                and len(g_label) == 1
                            ):
                                equal = p_label.upper() == g_label.upper()
                            # Else compare mapped texts if available
                            if not equal and p_text is not None and g_text is not None:
                                equal = p_text.strip().lower() == g_text.strip().lower()
                            # Fallback: direct string compare
                            if (
                                not equal
                                and isinstance(pred_ex, str)
                                and isinstance(gt_ex, str)
                            ):
                                equal = pred_ex.strip().lower() == gt_ex.strip().lower()

                            if equal and int(batch_verdicts[j]) == 0:
                                batch_verdicts[j] = 1
                                corrected = True
                                try:
                                    logger.warning(
                                        f"Judge corrected to 1 due to normalized equality (row={idx}).\nPrompt:\n{batch_prompts[j]}\nResponse:\n{resp}"
                                    )
                                except Exception:
                                    pass
                        if corrected:
                            logger.info(
                                "Applied post-judge equality correction for one or more items in batch."
                            )
                    except Exception:
                        pass
                    verdict_list.extend(batch_verdicts)

                    # Update progress
                    pbar.update(batch_end - i)

            data["verdict"] = verdict_list
            dump(data, result_path)

        # Load results and compute metrics
        data = load(result_path)

        # Calculate overall accuracy
        overall_acc = data["verdict"].mean() * 100

        # Create score summary
        score_df = pd.DataFrame(
            {"Metric": ["Overall Accuracy"], "Value": [overall_acc]}
        )

        # Save score summary
        dump(score_df, score_path)

        # Return results dictionary
        ret = {"Overall": overall_acc}

        # If there are categories in the data, compute per-category accuracy
        if "category" in data.columns:
            categories = data["category"].unique()
            for cat in categories:
                cat_data = data[data["category"] == cat]
                cat_acc = cat_data["verdict"].mean() * 100
                ret[cat] = cat_acc

        return ret
