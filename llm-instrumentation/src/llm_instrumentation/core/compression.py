from abc import ABC, abstractmethod
import lz4.frame
import zstandard as zstd
import torch
from typing import Tuple


class CompressionStrategy(ABC):
    @abstractmethod
    def compress(self, data: bytes) -> bytes: ...

    @abstractmethod
    def decompress(self, data: bytes) -> bytes: ...

    @abstractmethod
    def get_ratio(self) -> float: ...


class LZ4Strategy(CompressionStrategy):
    def __init__(self, compression_level: int = 1):
        self.compression_level = compression_level

    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data, compression_level=self.compression_level)

    def decompress(self, data: bytes) -> bytes:
        return lz4.frame.decompress(data)

    def get_ratio(self) -> float:
        return 0.0  # Placeholder


class ZstdStrategy(CompressionStrategy):
    def __init__(self, compression_level: int = 3):
        self.cctx = zstd.ZstdCompressor(level=compression_level)
        self.dctx = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        return self.cctx.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return self.dctx.decompress(data)

    def get_ratio(self) -> float:
        return 0.0  # Placeholder


class FP16Strategy(CompressionStrategy):
    def compress(self, data: bytes) -> bytes:
        raise NotImplementedError

    def decompress(self, data: bytes) -> bytes:
        raise NotImplementedError

    def get_ratio(self) -> float:
        raise NotImplementedError


class DeltaEncodingStrategy(CompressionStrategy):
    def compress(self, data: bytes) -> bytes:
        raise NotImplementedError

    def decompress(self, data: bytes) -> bytes:
        raise NotImplementedError

    def get_ratio(self) -> float:
        raise NotImplementedError


class TensorCompressionManager:
    """Adaptive tensor compression with multiple strategies."""

    def __init__(self):
        self.strategies = {
            "lz4": LZ4Strategy(),
            "zstd": ZstdStrategy(),
            "fp16": FP16Strategy(),
            "delta": DeltaEncodingStrategy(),
            "none": NoneStrategy(),
        }
        self.current_strategy = "lz4"

    def compress_tensor(self, tensor: torch.Tensor) -> Tuple[bytes, float]:
        """Compress tensor with automatic strategy selection."""

        # Convert to appropriate precision
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)  # FP32 -> FP16 compression

        # Serialize tensor
        tensor_bytes = self._tensor_to_bytes(tensor)

        # Apply compression
        compressed = self.strategies[self.current_strategy].compress(tensor_bytes)
        ratio = len(tensor_bytes) / len(compressed) if len(compressed) > 0 else 0

        return compressed, ratio

    def decompress_tensor(self, data: bytes, strategy: str = None) -> bytes:
        """Decompress tensor bytes using the selected strategy."""
        key = strategy or self.current_strategy
        if key not in self.strategies:
            raise ValueError(f"Unsupported compression strategy: {key}")
        decompressor = self.strategies[key]
        return decompressor.decompress(data)

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Efficient tensor serialization."""
        return tensor.cpu().numpy().tobytes()


class NoneStrategy(CompressionStrategy):
    """No-op compression strategy for raw byte passthrough."""

    def compress(self, data: bytes) -> bytes:
        return data

    def decompress(self, data: bytes) -> bytes:
        return data

    def get_ratio(self) -> float:
        return 1.0


# No-op compression strategy added directly to TensorCompressionManager
