"""
Utilities for collecting low-overhead runtime metrics via eBPF.

The ebpf package exposes composable collectors (disk, syscalls, memory, scheduler,
network) together with an interval-based aggregator and correlation helpers that
integrate with the rest of the llm-instrumentation stack.
"""

from .aggregator.interval import IntervalAggregator, AggregationConfig
from .collectors.disk_io import DiskIOCollector
from .collectors.syscalls_io import SyscallLatencyCollector
from .collectors.sync_wait import SyncWaitCollector
from .collectors.mem import MemoryPressureCollector
from .collectors.sched import SchedulerCollector
from .collectors.net import NetworkCollector

__all__ = [
    "AggregationConfig",
    "IntervalAggregator",
    "DiskIOCollector",
    "SyscallLatencyCollector",
    "MemoryPressureCollector",
    "SyncWaitCollector",
    "SchedulerCollector",
    "NetworkCollector",
]
