"""
Model detection utilities for automatically identifying model architectures from HuggingFace repositories.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from functools import partial

try:
    from transformers import AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def download_config(model_path: str) -> Dict[str, Any]:
    """Download config.json from HuggingFace repository."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests is required for model detection. Install with: pip install requests")

    # Handle both local paths and HF repository names
    if os.path.exists(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"config.json not found in {model_path}")

    # Try HuggingFace Hub API
    if '/' not in model_path:
        raise ValueError(f"Invalid model path: {model_path}. Must be either a local path or HF repo name "
                         f"like 'org/model'")

    config_url = f"https://huggingface.co/{model_path}/raw/main/config.json"

    try:
        response = requests.get(config_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to download config from {config_url}: {e}")


def detect_model_architecture(model_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Detect model architecture from HuggingFace repository.

    Args:
        model_path: Path to local model or HuggingFace repository name

    Returns:
        Tuple of (detected_class_name, default_config)
    """
    try:
        config = download_config(model_path)
    except Exception as e:
        raise ValueError(f"Could not load config for {model_path}: {e}")

    model_type = config.get("model_type", "").lower()
    architectures = config.get("architectures", [])

    # Convert architectures to lowercase for comparison
    arch_lower = [arch.lower() for arch in architectures]

    # Model detection logic based on config
    if (any("qwen2_vl" in arch or "qwen2_5_vl" in arch for arch in arch_lower) or
            model_type == "qwen2_vl" or model_type == "qwen2_5_vl"):
        # Handle both Qwen2-VL and Qwen2.5-VL models
        return "Qwen2VLChat", {
            "model_path": model_path,
            "min_pixels": 1280 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
            "use_custom_prompt": False,
        }

    elif any("qwen3_vl" in arch for arch in arch_lower) or model_type == "qwen3_vl":
        return "Qwen3VLChat", {
            "model_path": model_path,
            "min_pixels": 1280 * 28 * 28,
            "max_pixels": 16384 * 28 * 28,
            "use_custom_prompt": False,
        }

    elif any("qwenvl" in arch for arch in arch_lower) or model_type == "qwen_vl":
        if "chat" in model_path.lower():
            return "QwenVLChat", {"model_path": model_path}
        else:
            return "QwenVL", {"model_path": model_path}

    elif any("llava" in arch for arch in arch_lower) or "llava" in model_type:
        # Check for different LLaVA variants
        if "llava_next" in model_path.lower() or "llava-v1.6" in model_path.lower():
            return "LLaVA_Next", {"model_path": model_path}
        elif "onevision" in model_path.lower():
            if "hf" in model_path.lower():
                return "LLaVA_OneVision_HF", {"model_path": model_path}
            else:
                return "LLaVA_OneVision", {"model_path": model_path}
        else:
            return "LLaVA", {"model_path": model_path}

    elif any("internvl" in arch for arch in arch_lower) or "internvl" in model_type:
        # Determine InternVL version
        if "internvl3" in model_path.lower():
            return "InternVLChat", {"model_path": model_path, "version": "V2.0"}
        elif "internvl2" in model_path.lower():
            if "mpo" in model_path.lower():
                return "InternVLChat", {
                    "model_path": model_path,
                    "version": "V2.0",
                    "use_mpo_prompt": True
                }
            else:
                return "InternVLChat", {"model_path": model_path, "version": "V2.0"}
        else:
            # InternVL v1
            if "v1-5" in model_path.lower():
                return "InternVLChat", {"model_path": model_path, "version": "V1.5"}
            elif "v1-2" in model_path.lower():
                return "InternVLChat", {"model_path": model_path, "version": "V1.2"}
            else:
                return "InternVLChat", {"model_path": model_path, "version": "V1.1"}

    elif any("minicpm" in arch for arch in arch_lower) or "minicpm" in model_type:
        if "llama3" in model_path.lower():
            return "MiniCPM_Llama3_V", {"model_path": model_path}
        elif "2_6" in model_path.lower() or "2.6" in model_path.lower():
            if "o-2_6" in model_path.lower() or "o-2.6" in model_path.lower():
                return "MiniCPM_o_2_6", {"model_path": model_path}
            else:
                return "MiniCPM_V_2_6", {"model_path": model_path}
        else:
            return "MiniCPM_V", {"model_path": model_path}

    elif any("phi" in arch for arch in arch_lower) or "phi" in model_type:
        if "phi-4" in model_path.lower() or config.get("_name_or_path", "").startswith("microsoft/Phi-4"):
            return "Phi4Multimodal", {"model_path": model_path}
        elif "phi-3.5" in model_path.lower():
            return "Phi3_5Vision", {"model_path": model_path}
        elif "phi-3" in model_path.lower():
            return "Phi3Vision", {"model_path": model_path}

    elif any("molmo" in arch for arch in arch_lower) or "molmo" in model_type:
        return "molmo", {"model_path": model_path}

    elif any("aria" in arch for arch in arch_lower) or "aria" in model_type:
        return "Aria", {"model_path": model_path}

    elif any("pixtral" in arch for arch in arch_lower) or "pixtral" in model_type:
        return "Pixtral", {"model_path": model_path}

    elif any("smolvlm" in arch for arch in arch_lower) or "smolvlm" in model_type:
        if "smolvlm2" in model_path.lower():
            return "SmolVLM2", {"model_path": model_path}
        else:
            return "SmolVLM", {"model_path": model_path}

    elif any("idefics" in arch for arch in arch_lower) or "idefics" in model_type:
        if "idefics3" in model_path.lower() or "idefics2" in model_path.lower():
            return "IDEFICS2", {"model_path": model_path}
        else:
            return "IDEFICS", {"model_path": model_path}

    elif any("cogvlm" in arch for arch in arch_lower) or "cogvlm" in model_type:
        if "glm-4v" in model_path.lower():
            return "GLM4v", {"model_path": model_path}
        else:
            return "CogVlm", {"model_path": model_path}

    elif any("deepseek" in arch for arch in arch_lower) or "deepseek" in model_type:
        if "janus" in model_path.lower():
            return "Janus", {"model_path": model_path}
        elif "vl2" in model_path.lower():
            return "DeepSeekVL2", {"model_path": model_path}
        else:
            return "DeepSeekVL", {"model_path": model_path}

    elif any("llama" in arch for arch in arch_lower) and "vision" in model_path.lower():
        if "llama-4" in model_path.lower():
            return "llama4", {"model_path": model_path, "use_vllm": True}
        else:
            return "llama_vision", {"model_path": model_path}

    elif any("gemma" in arch for arch in arch_lower) or "gemma" in model_type:
        if "paligemma" in model_path.lower():
            return "PaliGemma", {"model_path": model_path}
        else:
            return "Gemma3", {"model_path": model_path}

    # Fallback: try some common patterns based on model name
    model_name_lower = model_path.lower()

    if "vila" in model_name_lower:
        if "nvila" in model_name_lower:
            return "NVILA", {"model_path": model_path}
        else:
            return "VILA", {"model_path": model_path}

    elif "ovis" in model_name_lower:
        if "ovis2" in model_name_lower:
            return "Ovis2", {"model_path": model_path}
        elif "ovis1.6" in model_name_lower:
            if "27b" in model_name_lower:
                return "Ovis1_6_Plus", {"model_path": model_path}
            else:
                return "Ovis1_6", {"model_path": model_path}
        else:
            return "Ovis", {"model_path": model_path}

    elif "bunny" in model_name_lower:
        if "llama3" in model_name_lower or "llama-3" in model_name_lower:
            return "BunnyLLama3", {"model_path": model_path}
        # Also check model type for bunny
    elif model_type == "bunny-llama":
        return "BunnyLLama3", {"model_path": model_path}

    elif "cambrian" in model_name_lower:
        return "Cambrian", {"model_path": model_path}

    elif "mantis" in model_name_lower:
        return "Mantis", {"model_path": model_path}

    elif "moondream" in model_name_lower:
        if "moondream2" in model_name_lower:
            return "Moondream2", {"model_path": model_path}
        else:
            return "Moondream1", {"model_path": model_path}

    elif "eagle" in model_name_lower:
        return "Eagle", {"model_path": model_path}

    elif "vita" in model_name_lower:
        if "long" in model_name_lower:
            return "LongVITA", {"model_path": model_path}
        elif "qwen2" in model_name_lower:
            return "VITAQwen2", {"model_path": model_path}
        else:
            return "VITA", {"model_path": model_path}

    elif "sail" in model_name_lower:
        return "SailVL", {"model_path": model_path}

    elif "flash" in model_name_lower and "vl" in model_name_lower:
        return "FlashVL", {"model_path": model_path}

    elif "kimi" in model_name_lower:
        return "KimiVL", {"model_path": model_path}

    elif "nvlm" in model_name_lower:
        return "NVLM", {"model_path": model_path}

    elif "vintern" in model_name_lower:
        return "VinternChat", {"model_path": model_path}

    elif "h2ovl" in model_name_lower:
        return "H2OVLChat", {"model_path": model_path}

    elif "points" in model_name_lower:
        if "v15" in model_name_lower or "v1.5" in model_name_lower:
            return "POINTSV15", {"model_path": model_path}
        else:
            return "POINTS", {"model_path": model_path}

    elif "kosmos" in model_name_lower:
        return "Kosmos2", {"model_path": model_path}

    elif "emu" in model_name_lower:
        if "emu3" in model_name_lower:
            if "gen" in model_name_lower:
                return "Emu3_gen", {"model_path": model_path}
            else:
                return "Emu3_chat", {"model_path": model_path}
        else:
            return "Emu", {"model_path": model_path}

    # If no match found, raise an error with helpful information
    supported_types = [
        "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen_vl", "llava", "internvl", "minicpm", "phi",
        "molmo", "aria", "pixtral", "smolvlm", "idefics", "cogvlm",
        "deepseek", "llama-vision", "gemma", "vila", "ovis", "bunny",
        "cambrian", "mantis", "moondream", "eagle", "vita", "sail", "flash",
        "kimi", "nvlm", "vintern", "h2ovl", "points", "kosmos", "emu"
    ]

    raise ValueError(
        f"Could not auto-detect model architecture for {model_path}.\n"
        f"Model type: {model_type}\n"
        f"Architectures: {architectures}\n"
        f"Supported model types: {supported_types}\n"
        f"Please manually specify the model in the config or add detection logic for this model type."
    )


def create_custom_model_entry(model_path: str, model_name: Optional[str] = None) -> Tuple[str, partial]:
    """
    Create a custom model entry for the supported_VLM dictionary.

    Args:
        model_path: Path to model or HuggingFace repository name
        model_name: Optional custom name for the model (defaults to path basename)

    Returns:
        Tuple of (model_name, partial_function)
    """
    detected_class, config = detect_model_architecture(model_path)

    if model_name is None:
        # Replace slashes with underscores to preserve org name and avoid directory issues
        model_name = model_path.replace('/', '_') if '/' in model_path else model_path

    # Import the detected class
    try:
        import vlmeval.vlm
        import vlmeval.api

        if hasattr(vlmeval.vlm, detected_class):
            model_class = getattr(vlmeval.vlm, detected_class)
        elif hasattr(vlmeval.api, detected_class):
            model_class = getattr(vlmeval.api, detected_class)
        else:
            raise ImportError(f"Model class {detected_class} not found in vlmeval.vlm or vlmeval.api")

        return model_name, partial(model_class, **config)

    except ImportError as e:
        raise ImportError(f"Failed to import model class {detected_class}: {e}")


def is_vllm_compatible(detected_class: str, model_path: str) -> bool:
    """
    Determine if a model class/path combination is VLLM compatible.

    Args:
        detected_class: The detected model class name
        model_path: The model path or repository name

    Returns:
        True if the model is VLLM compatible
    """
    # Check based on model class
    vllm_compatible_classes = {
        "Qwen2VLChat",  # Qwen2-VL and Qwen2.5-VL models
        "Qwen3VLChat",  # Qwen3-VL models
        "llama4",       # Llama-4 models
        "molmo",        # Molmo models
        "Gemma3",       # Gemma models (in some configurations)
    }

    if detected_class in vllm_compatible_classes:
        return True

    # Additional path-based checks for edge cases
    model_path_lower = model_path.lower()
    if any(pattern in model_path_lower for pattern in ["qwen2-vl", "qwen2.5-vl", "llama-4", "molmo"]):
        return True

    return False


# Global set to track VLLM-compatible custom models
_vllm_compatible_models = set()


def register_custom_model(model_path: str, model_name: Optional[str] = None) -> str:
    """
    Register a custom model with VLMEvalKit's supported_VLM dictionary.

    Args:
        model_path: Path to model or HuggingFace repository name
                   If prefixed with "/LOCAL_MODEL", the remainder is treated as an absolute local path
        model_name: Optional custom name for the model

    Returns:
        The registered model name
    """
    from ..config import supported_VLM

    # Handle /LOCAL_MODEL prefix
    if model_path.startswith("/LOCAL_MODEL"):
        # Extract the actual path after the prefix
        actual_path = model_path[len("/LOCAL_MODEL"):]
        # Ensure it's an absolute path
        if not os.path.isabs(actual_path):
            raise ValueError(f"Path after /LOCAL_MODEL must be absolute, got: {actual_path}")
        if not os.path.exists(actual_path):
            raise ValueError(f"Local model path does not exist: {actual_path}")
        if not os.path.isdir(actual_path):
            raise ValueError(f"Local model path must be a directory, got: {actual_path}")
        # Use the actual path for model detection
        model_path = actual_path

    registered_name, model_partial = create_custom_model_entry(model_path, model_name)

    # Add to supported_VLM dictionary
    supported_VLM[registered_name] = model_partial

    # Store VLLM compatibility information for later use
    detected_class, _ = detect_model_architecture(model_path)
    if is_vllm_compatible(detected_class, model_path):
        # Add to global VLLM compatibility tracker
        _vllm_compatible_models.add(registered_name)

    return registered_name


def is_model_vllm_compatible(model_name: str) -> bool:
    """
    Check if a model (including custom registered models) is VLLM compatible.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is VLLM compatible
    """
    # Check traditional hardcoded patterns
    is_vllm_compatible_traditional = (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen3-VL' in model_name
        or 'molmo' in model_name.lower()
    )

    # Also check if this is a custom registered model with VLLM compatibility
    is_custom_vllm_compatible = model_name in _vllm_compatible_models

    return is_vllm_compatible_traditional or is_custom_vllm_compatible
