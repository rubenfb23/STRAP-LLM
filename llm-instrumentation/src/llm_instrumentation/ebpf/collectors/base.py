"""
Base classes for eBPF collectors.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

from ..bpf_loader import BPFHandle, load_bpf_module
from ..exceptions import CollectorAttachError

LOG = logging.getLogger(__name__)


def _resolve_cgroup_id(path: str) -> Optional[int]:
    from pathlib import Path

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path("/sys/fs/cgroup") / candidate
    try:
        stat = candidate.stat()
        return stat.st_ino
    except FileNotFoundError:
        return None


@dataclass
class CollectorConfig:
    """Common configuration shared by all collectors."""

    pid: Optional[int] = None
    cgroup_path: Optional[str] = None
    comm_filter: Optional[str] = None
    sample_every: int = 1
    bcc_paths: tuple[str, ...] = ()
    cgroup_id: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cgroup_path:
            self.cgroup_id = _resolve_cgroup_id(self.cgroup_path)

    def cflags(self) -> list[str]:
        flags: list[str] = []
        if self.pid:
            flags.append(f"-DTARGET_PID={self.pid}")
        if self.comm_filter:
            flags.append(f'-DTARGET_COMM="{self.comm_filter}"')
        if self.cgroup_id:
            flags.append(f"-DTARGET_CGROUP_ID={self.cgroup_id}")
        if self.sample_every and self.sample_every > 1:
            flags.append(f"-DSAMPLE_EVERY={self.sample_every}")
        return flags


class BaseCollector(abc.ABC):
    """Contract for kernel-space collectors."""

    NAME: str = ""

    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self._bpf_handle: Optional[BPFHandle] = None
        self._bpf: Optional[Any] = None
        self._attached = False

    @property
    def bpf(self) -> Any:
        if self._bpf is None:
            raise RuntimeError("Collector not attached yet")
        return self._bpf

    def ensure_loaded(self) -> Any:
        if self._bpf is not None:
            return self._bpf
        paths = tuple(
            path
            for path in self.config.bcc_paths
            if path
        )
        self._bpf_handle = load_bpf_module(extra_paths=tuple(map(str, paths)))
        text = self.build_bpf()
        self._bpf = self._bpf_handle.new(text=text, cflags=self.config.cflags())
        return self._bpf

    def attach(self) -> None:
        if self._attached:
            return
        try:
            self.do_attach(self.ensure_loaded())
        except Exception as exc:  # pragma: no cover - relies on kernel availability
            raise CollectorAttachError(f"Failed to attach collector {self.NAME}: {exc}") from exc
        self._attached = True
        LOG.debug("Collector %s attached", self.NAME)

    def detach(self) -> None:
        try:
            if self._bpf and hasattr(self._bpf, "detach_kprobe"):
                # BCC will automatically detach on destruction, but be explicit
                pass
        finally:
            self._attached = False

    @abc.abstractmethod
    def build_bpf(self) -> str:
        """Return the BPF program for this collector."""

    @abc.abstractmethod
    def do_attach(self, bpf: Any) -> None:
        """Attach tracepoints/kprobes/uprobes."""

    @abc.abstractmethod
    def consume(self) -> Iterator[Dict[str, Any]]:
        """Yield raw events to be aggregated."""

    def reset(self) -> None:
        """Clear kernel maps if needed."""
        pass


def decode_comm(raw: Iterable[int]) -> str:
    bytes_ = bytes(raw)
    return bytes_.split(b"\x00", 1)[0].decode("utf-8", "replace")
