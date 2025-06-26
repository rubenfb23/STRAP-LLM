import time
import torch
from dataclasses import dataclass, field
from typing import List, Optional
import threading
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking."""

    inference_times: List[float] = field(default_factory=list)
    compression_ratios: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    throughput_mbps: List[float] = field(default_factory=list)
    buffer_overruns: int = 0
    total_tokens_processed: int = 0

    def record_inference(self, duration: float) -> None:
        """Record inference timing."""
        self.inference_times.append(duration)

    def record_compression_ratio(self, ratio: float) -> None:
        """Record compression ratio."""
        self.compression_ratios.append(ratio)

    def record_throughput(self, throughput: float) -> None:
        """Record throughput."""
        self.throughput_mbps.append(throughput)

    def increment_buffer_overruns(self) -> None:
        """Increment buffer overruns counter."""
        self.buffer_overruns += 1

    def add_processed_tokens(self, count: int) -> None:
        """Add to total processed tokens."""
        self.total_tokens_processed += count

    def get_average_throughput(self) -> float:
        """Calculate average throughput degradation."""
        if not self.inference_times:
            return 0.0
        return 1.0 / (sum(self.inference_times) / len(self.inference_times))


class SystemProfiler:
    """Real-time system performance monitoring."""

    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.metrics = PerformanceMetrics()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics

    def _monitor_loop(self) -> None:
        while self.monitoring:
            # This is a placeholder for actual monitoring logic
            time.sleep(self.sampling_interval)

    @contextmanager
    def measure_inference(self):
        """Context manager for measuring inference performance."""
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        yield

        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        self.metrics.record_inference(end_time - start_time)
        self.metrics.memory_usage.append(end_memory - start_memory)

    def record_compression_ratio(self, ratio: float) -> None:
        """Delegate recording compression ratio to metrics."""
        self.metrics.record_compression_ratio(ratio)

    def record_throughput(self, throughput: float) -> None:
        """Delegate recording throughput to metrics."""
        self.metrics.record_throughput(throughput)

    def increment_buffer_overruns(self) -> None:
        """Delegate incrementing buffer overruns to metrics."""
        self.metrics.increment_buffer_overruns()

    def add_processed_tokens(self, count: int) -> None:
        """Delegate adding processed tokens to metrics."""
        self.metrics.add_processed_tokens(count)
