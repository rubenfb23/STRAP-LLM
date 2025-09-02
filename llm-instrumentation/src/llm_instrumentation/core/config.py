from dataclasses import dataclass
from typing import Optional
from .hooks import HookGranularity


@dataclass
class InstrumentationConfig:
    """Configuration for LLM instrumentation framework."""

    granularity: HookGranularity
    compression_algorithm: str = "lz4"
    target_throughput_gbps: float = 1.0
    max_memory_gb: Optional[float] = None

    def __post_init__(self):
        if self.compression_algorithm not in ["lz4", "zstd", "none"]:
            raise ValueError("Compression algorithm must be 'lz4', 'zstd', or 'none'")
