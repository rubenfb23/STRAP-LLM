import pytest
import time
from llm_instrumentation.core.profiling import PerformanceMetrics, SystemProfiler


def test_performance_metrics_initialization():
    """Test that PerformanceMetrics initializes with empty lists and zero counters."""
    metrics = PerformanceMetrics()
    assert metrics.inference_times == []
    assert metrics.compression_ratios == []
    assert metrics.memory_usage == []
    assert metrics.throughput_mbps == []
    assert metrics.buffer_overruns == 0
    assert metrics.total_tokens_processed == 0


def test_record_inference():
    """Test recording of inference time."""
    metrics = PerformanceMetrics()
    metrics.record_inference(0.123)
    assert metrics.inference_times == [0.123]


def test_record_compression_ratio():
    """Test recording of compression ratio."""
    metrics = PerformanceMetrics()
    metrics.record_compression_ratio(1.5)
    assert metrics.compression_ratios == [1.5]


def test_record_throughput():
    """Test recording of throughput."""
    metrics = PerformanceMetrics()
    metrics.record_throughput(100.5)
    assert metrics.throughput_mbps == [100.5]


def test_increment_buffer_overruns():
    """Test incrementing buffer overruns."""
    metrics = PerformanceMetrics()
    metrics.increment_buffer_overruns()
    assert metrics.buffer_overruns == 1
    metrics.increment_buffer_overruns()
    assert metrics.buffer_overruns == 2


def test_add_processed_tokens():
    """Test adding processed tokens."""
    metrics = PerformanceMetrics()
    metrics.add_processed_tokens(128)
    assert metrics.total_tokens_processed == 128
    metrics.add_processed_tokens(256)
    assert metrics.total_tokens_processed == 384


def test_get_average_throughput():
    """Test calculation of average throughput."""
    metrics = PerformanceMetrics()
    assert metrics.get_average_throughput() == 0.0
    metrics.record_inference(0.2)
    metrics.record_inference(0.3)
    # Average inference time = 0.25s. Throughput = 1/0.25 = 4 inferences/sec
    assert metrics.get_average_throughput() == pytest.approx(1 / 0.25)


def test_system_profiler_measure_inference():
    """Test the measure_inference context manager."""
    profiler = SystemProfiler()
    with profiler.measure_inference():
        time.sleep(0.01)

    assert len(profiler.metrics.inference_times) == 1
    assert profiler.metrics.inference_times[0] > 0.009
    assert len(profiler.metrics.memory_usage) == 1


def test_system_profiler_monitoring():
    """Test the background monitoring thread."""
    profiler = SystemProfiler(sampling_interval=0.01)
    profiler.start_monitoring()
    assert profiler.monitoring
    assert profiler.monitor_thread.is_alive()
    time.sleep(0.05)
    metrics = profiler.stop_monitoring()
    assert not profiler.monitoring
    # Give thread time to stop
    time.sleep(0.02)
    assert not profiler.monitor_thread.is_alive()
    assert isinstance(metrics, PerformanceMetrics)


def test_system_profiler_delegation():
    """Test that SystemProfiler correctly delegates calls to PerformanceMetrics."""
    profiler = SystemProfiler()
    profiler.record_compression_ratio(1.5)
    profiler.record_throughput(100.5)
    profiler.increment_buffer_overruns()
    profiler.add_processed_tokens(128)

    metrics = profiler.metrics
    assert metrics.compression_ratios == [1.5]
    assert metrics.throughput_mbps == [100.5]
    assert metrics.buffer_overruns == 1
    assert metrics.total_tokens_processed == 128
