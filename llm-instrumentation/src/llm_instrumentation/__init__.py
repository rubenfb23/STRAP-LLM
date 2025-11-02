"""
STRAP-LLM: Streaming Transformer Activation Pipeline

A high-performance instrumentation framework for capturing, streaming, and analyzing
internal activations of large language models (LLMs).
"""

from llm_instrumentation.core.framework import (
    InstrumentationFramework,
    analyze_activations_with_tokens,
)
from llm_instrumentation.core.config import InstrumentationConfig
from llm_instrumentation.core.hooks import HookGranularity, HookConfig
from llm_instrumentation.core.streaming import StreamingSerializer
from llm_instrumentation.core.compression import TensorCompressionManager

__version__ = "0.1.0"

__all__ = [
    "InstrumentationFramework",
    "InstrumentationConfig",
    "HookGranularity",
    "HookConfig",
    "StreamingSerializer",
    "TensorCompressionManager",
    "analyze_activations_with_tokens",
    "__version__",
]
