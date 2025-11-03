import os
from contextlib import contextmanager
from datetime import datetime
from typing import Literal, Optional

from .config import HookGranularity, InstrumentationConfig
from .framework import InstrumentationFramework


def detect_model_architecture(model) -> str:
    """Detect the model architecture type.

    Args:
        model: PyTorch model.

    Returns:
        Architecture string ('gpt', 'llama', 'qwen', 'bert', 'unknown').
    """
    model_type = "unknown"

    if hasattr(model, "config"):
        if hasattr(model.config, "model_type"):
            model_type = str(model.config.model_type).lower()
        elif hasattr(model.config, "architectures"):
            arch = str(model.config.architectures[0]).lower()
            if "gpt" in arch:
                model_type = "gpt"
            elif "llama" in arch:
                model_type = "llama"
            elif "qwen" in arch:
                model_type = "qwen"
            elif "bert" in arch:
                model_type = "bert"

    if model_type == "unknown" and hasattr(model, "__class__"):
        class_name = model.__class__.__name__.lower()
        for keyword in ["gpt", "llama", "qwen", "bert"]:
            if keyword in class_name:
                model_type = keyword
                break

    return model_type


def create_optimized_config(
    model_arch: str,
    purpose: Literal["interpretability", "debugging", "performance_analysis", "full_capture"],
) -> InstrumentationConfig:
    """Create a purpose-optimized configuration based on architecture."""
    if purpose == "interpretability":
        return InstrumentationConfig.attention_analysis()
    if purpose == "debugging":
        return InstrumentationConfig.fast_capture()
    if purpose == "performance_analysis":
        base_config = InstrumentationConfig.balanced()
        if model_arch in {"gpt", "llama", "qwen"}:
            return base_config.with_compression("zstd").with_memory_limit(16)
        return base_config
    if purpose == "full_capture":
        return InstrumentationConfig.balanced()
    return InstrumentationConfig.balanced()

