from __future__ import annotations

import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory


VLLM_MAX_IMAGE_INPUT_NUM = 24


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model."""
    # First check config.json for num_experts (most reliable)
    import json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            num_experts = cfg.get("num_experts", 0)
            if isinstance(num_experts, int):
                return num_experts > 0
        except Exception:
            pass
    # Fall back to path-name heuristic
    path_parts = model_path.split("/")
    non_moe_patterns = ["4B", "8B", "4b", "8b"]
    for part in path_parts:
        if any(pattern in part for pattern in non_moe_patterns):
            return False
    return True


def ensure_image_url(image: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:image"]
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return "file://" + image
    raise ValueError(f"Invalid image: {image}")


def ensure_video_url(video: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:video"]
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return "file://" + video
    raise ValueError(f"Invalid video: {video}")


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = 1.5
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print(
                "The total number of video tokens might too large, resulting in an overly long input sequence."
            )
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop("fps", 2)
        self.nframe = kwargs.pop("nframe", 128)
        self.FRAME_FACTOR = 2

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get("use_vllm", False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.use_vllm:
            from vllm import LLM

            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f"Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})"
            )
            if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=self.max_new_tokens,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                mm_encoder_tp_mode="data",
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.7),
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 32 * 32
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                for key in [
                    "min_pixels",
                    "max_pixels",
                    "total_pixels",
                    "resized_height",
                    "resized_width",
                ]:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s["type"] == "video":
                value = s["value"]
                if isinstance(value, list):
                    item = {
                        "type": "video",
                        "video": [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {"type": "video", "video": ensure_video_url(value)}
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                for key in [
                    "resized_height",
                    "resized_width",
                    "fps",
                    "nframes",
                    "sample_fps",
                ]:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and "fps" not in item:
                        item["fps"] = self.fps
                    elif self.nframe is not None and "nframes" not in item:
                        import cv2

                        video = cv2.VideoCapture(s["value"])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = (
                                frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            )
                            print(f"use {new_frame_count} for {s['value']}")
                            item["nframes"] = new_frame_count
                        else:
                            item["nframes"] = self.nframe
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_vllm(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """vLLM-safe variant that enforces modality limits."""
        content: list[dict[str, str]] = []
        video_inputs = [s for s in inputs if s["type"] == "video"]
        video_count = len(video_inputs)
        cur_image_count = 0

        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 32 * 32
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                for key in [
                    "min_pixels",
                    "max_pixels",
                    "total_pixels",
                    "resized_height",
                    "resized_width",
                ]:
                    if key in s and s[key] is not None:
                        item[key] = s[key]

                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        "Image count exceeds vLLM limit (%d); dropping extra images.",
                        self.limit_mm_per_prompt,
                    )
            elif s["type"] == "video":
                value = s["value"]
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected for vLLM; only the first will be used."
                    )
                    if s is not video_inputs[0]:
                        continue

                if isinstance(value, list):
                    item = {
                        "type": "video",
                        "video": [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {"type": "video", "video": ensure_video_url(value)}

                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.total_pixels is not None:
                    item["total_pixels"] = self.total_pixels
                for key in [
                    "resized_height",
                    "resized_width",
                    "fps",
                    "nframes",
                    "sample_fps",
                ]:
                    if key in s and s[key] is not None:
                        item[key] = s[key]

                if not isinstance(value, list):
                    if self.fps is not None and "fps" not in item:
                        item["fps"] = self.fps
                    elif self.nframe is not None and "nframes" not in item:
                        import cv2

                        video = cv2.VideoCapture(s["value"])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = (
                                frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            )
                            print(f"use {new_frame_count} for {s['value']}")
                            item["nframes"] = new_frame_count
                        else:
                            item["nframes"] = self.nframe
                content.append(item)
            elif s["type"] == "text":
                content.append({"type": "text", "text": s["value"]})
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")

        return content

    def generate_inner_transformers(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self._prepare_content_vllm(message, dataset=dataset),
            }
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        req = {"prompt": text}
        if mm_data:
            req["multi_modal_data"] = mm_data
        if video_kwargs is not None:
            req["mm_processor_kwargs"] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f"\033[32m{generated_text}\033[0m")
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)

    # =============================================================================
    # BATCH PROCESSING METHODS (VLLM-ONLY)
    # =============================================================================

    def supports_batch_processing(self) -> bool:
        """Check if this model instance supports batch processing."""
        return self.use_vllm

    def get_optimal_batch_size(self, estimated_items: int | None = None) -> int:
        """Estimate a safe batch size for the current configuration."""
        if not self.use_vllm:
            return 1

        # vLLM for Qwen3-VL defaults to a smaller max_num_seqs (5); stay conservative.
        base_batch_size = 5
        try:
            llm_engine = getattr(self.llm, "llm_engine", None)
            if llm_engine is not None:
                scheduler_cfg = getattr(llm_engine, "scheduler_config", None)
                inferred = getattr(scheduler_cfg, "max_num_seqs", None)
                if isinstance(inferred, int) and inferred > 0:
                    base_batch_size = inferred
        except Exception:
            # Fall back to conservative default if inspection fails.
            pass

        if estimated_items is not None:
            return max(1, min(estimated_items, base_batch_size))

        return base_batch_size

    def generate_batch_vllm(
        self,
        batch_messages: list,
        dataset: str | None = None,
        batch_size: int | None = None,
    ) -> list:
        """Generate responses for a batch of messages using vLLM."""
        if not self.use_vllm:
            raise ValueError("Batch processing requires use_vllm=True")

        if not batch_messages:
            return []

        if batch_size is None:
            batch_size = min(len(batch_messages), self.get_optimal_batch_size())
        else:
            batch_size = max(1, min(batch_size, self.get_optimal_batch_size()))

        # Sequential path when effective batch size is 1.
        if batch_size == 1 or len(batch_messages) == 1:
            return [
                self.generate_inner_vllm(msg, dataset=dataset) for msg in batch_messages
            ]

        all_results: list = []

        for start in range(0, len(batch_messages), batch_size):
            chunk = batch_messages[start : start + batch_size]
            try:
                chunk_results = self._process_qwen_vllm_batch(
                    chunk, dataset=dataset
                )
                all_results.extend(chunk_results)
            except Exception as exc:
                if self.verbose:
                    logging.error(
                        "Batch processing failed for items %d-%d, fallback to sequential: %s",
                        start + 1,
                        start + len(chunk),
                        exc,
                    )
                for msg in chunk:
                    try:
                        all_results.append(
                            self.generate_inner_vllm(msg, dataset=dataset)
                        )
                    except Exception as seq_exc:
                        if self.verbose:
                            logging.error(
                                "Sequential fallback failed: %s", seq_exc
                            )
                        all_results.append("ERROR: Generation failed")

        return all_results

    def _process_qwen_vllm_batch(
        self,
        batch_messages: list,
        dataset: str | None = None,
    ) -> list:
        """Process a single batch through vLLM for Qwen3-VL."""
        from vllm import SamplingParams

        if not batch_messages:
            return []

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        vllm_inputs = []

        for idx, message in enumerate(batch_messages):
            try:
                messages = []
                if self.system_prompt is not None:
                    messages.append(
                        {"role": "system", "content": self.system_prompt}
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": self._prepare_content_vllm(
                            message, dataset=dataset
                        ),
                    }
                )

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )

                req = {"prompt": text}
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                if mm_data:
                    req["multi_modal_data"] = mm_data
                if video_kwargs:
                    req["mm_processor_kwargs"] = video_kwargs

                vllm_inputs.append(req)
            except Exception as prep_err:
                if self.verbose:
                    logging.warning(
                        "Failed to prepare vLLM input for batch item %d: %s",
                        idx,
                        prep_err,
                    )
                vllm_inputs.append({"prompt": "Error in input preparation"})

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None,
        )

        if self.verbose:
            logging.info(
                "[Qwen3-VL vLLM Batch] Processing %d items", len(vllm_inputs)
            )

        outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)

        results = []
        for idx, output in enumerate(outputs):
            try:
                if not output.outputs:
                    results.append("")
                    continue

                generated_text = output.outputs[0].text

                if self.post_process:
                    resp = generated_text.split("\\boxed{")[-1]
                    lt = len(resp)
                    counter, end = 1, None
                    for i in range(lt):
                        if resp[i] == "{":
                            counter += 1
                        elif resp[i] == "}":
                            counter -= 1
                        if counter == 0:
                            end = i
                            break
                        elif i == lt - 1:
                            end = lt
                            break
                    if end is not None:
                        generated_text = resp[:end]

                results.append(generated_text)

                if self.verbose:
                    logging.info(
                        "[Qwen3-VL vLLM Batch] Item %d: %.50s",
                        idx,
                        generated_text,
                    )
            except Exception as post_err:
                if self.verbose:
                    logging.warning(
                        "Failed to process output for batch item %d: %s",
                        idx,
                        post_err,
                    )
                results.append("ERROR: Output processing failed")

        return results
