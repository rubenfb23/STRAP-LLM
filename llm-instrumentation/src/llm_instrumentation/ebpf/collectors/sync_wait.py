from __future__ import annotations

from .base import CollectorConfig
from .syscalls_io import SyscallLatencyCollector


SYNC_WAIT_SYSCALLS = [
    "futex",
    "epoll_wait",
    "poll",
    "ppoll",
    "clock_nanosleep",
]


class SyncWaitCollector(SyscallLatencyCollector):
    """Specialized syscall collector focused on wait/synchronization calls."""

    NAME = "sync_wait"

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__(config, tracked_names=SYNC_WAIT_SYSCALLS)
