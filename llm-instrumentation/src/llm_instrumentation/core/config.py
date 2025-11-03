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

    @classmethod
    def fast_capture(cls) -> "InstrumentationConfig":
        """Fast capture with minimal overhead (no compression)."""
        return cls(
            granularity=HookGranularity.FULL_TENSOR,
            compression_algorithm="none",
            target_throughput_gbps=4.0,
            max_memory_gb=32,
        )

    @classmethod
    def balanced(cls) -> "InstrumentationConfig":
        """Balanced speed and compression (LZ4)."""
        return cls(
            granularity=HookGranularity.FULL_TENSOR,
            compression_algorithm="lz4",
            target_throughput_gbps=2.0,
            max_memory_gb=24,
        )

    @classmethod
    def max_compression(cls) -> "InstrumentationConfig":
        """Maximum compression (Zstd) with reduced throughput."""
        return cls(
            granularity=HookGranularity.FULL_TENSOR,
            compression_algorithm="zstd",
            target_throughput_gbps=1.0,
            max_memory_gb=16,
        )

    @classmethod
    def max(cls) -> "InstrumentationConfig":
        """Highest fidelity capture with maximum buffering headroom."""
        return cls(
            granularity=HookGranularity.FULL_TENSOR,
            compression_algorithm="zstd",
            target_throughput_gbps=6.0,
            max_memory_gb=64,
        )

    @classmethod
    def attention_analysis(cls) -> "InstrumentationConfig":
        """Attention-only capture for interpretability analysis."""
        return cls(
            granularity=HookGranularity.ATTENTION_ONLY,
            compression_algorithm="lz4",
            target_throughput_gbps=2.0,
            max_memory_gb=16,
        )

    @classmethod
    def mlp_analysis(cls) -> "InstrumentationConfig":
        """MLP-only capture for feature analysis."""
        return cls(
            granularity=HookGranularity.MLP_ONLY,
            compression_algorithm="lz4",
            target_throughput_gbps=2.0,
            max_memory_gb=16,
        )

    def with_compression(self, algo: str) -> "InstrumentationConfig":
        """Override compression algorithm.

        Args:
            algo: Compression algorithm ('none', 'lz4', 'zstd')

        Returns:
            Self for method chaining
        """
        self.compression_algorithm = algo
        return self

    def with_memory_limit(self, gb: float) -> "InstrumentationConfig":
        """Override memory limit.

        Args:
            gb: Memory limit in gigabytes

        Returns:
            Self for method chaining
        """
        self.max_memory_gb = gb
        return self

    def with_throughput(self, gbps: float) -> "InstrumentationConfig":
        """Override target throughput.

        Args:
            gbps: Target throughput in GB/s

        Returns:
            Self for method chaining
        """
        self.target_throughput_gbps = gbps
        return self

    def with_granularity(self, granularity: HookGranularity) -> "InstrumentationConfig":
        """Override hook granularity.

        Args:
            granularity: HookGranularity enum value

        Returns:
            Self for method chaining
        """
        self.granularity = granularity
        return self
