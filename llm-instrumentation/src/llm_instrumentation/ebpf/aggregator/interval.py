"""
Interval-based aggregator that reduces events coming from collectors.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence

from ..collectors import BaseCollector
from ..correlation.context import CorrelationContext


@dataclass
class AggregationConfig:
    interval: float = 1.0
    output_path: Optional[Path] = None
    flush_every: int = 5
    fsync: bool = False
    include_wall_time: bool = True
    extra_labels: Dict[str, Any] = field(default_factory=dict)


class IntervalAggregator:
    def __init__(
        self,
        collectors: Sequence[BaseCollector],
        config: AggregationConfig,
        corr_context: Optional[CorrelationContext] = None,
    ) -> None:
        self.collectors = list(collectors)
        self.config = config
        self.corr_context = corr_context or CorrelationContext.disabled()
        self._buffer: List[str] = []
        self._last_flush = time.monotonic()

        if self.config.output_path:
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.config.output_path.open("a", encoding="utf-8")
        else:
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._file:
            if self._buffer:
                self._flush()
            self._file.close()
            self._file = None

    def run_forever(self) -> None:
        for collector in self.collectors:
            collector.attach()
        try:
            while True:
                start = time.monotonic()
                events = list(self._collect_once())
                if events:
                    self._emit(events)
                elapsed = time.monotonic() - start
                sleep_for = max(0.0, self.config.interval - elapsed)
                if sleep_for:
                    time.sleep(sleep_for)
        finally:
            for collector in self.collectors:
                collector.detach()
            self.close()

    def _collect_once(self) -> Iterator[Dict[str, Any]]:
        now_ns = time.time_ns()
        wall_ts = (
            datetime.now(timezone.utc).isoformat()
            if self.config.include_wall_time
            else None
        )
        for collector in self.collectors:
            for event in collector.consume():
                event.setdefault("ts_ns", now_ns)
                if wall_ts:
                    event.setdefault("ts", wall_ts)
                event.setdefault("type", collector.NAME)
                if self.corr_context.enabled:
                    event.setdefault("corr_id", self.corr_context.current_id())
                if self.config.extra_labels:
                    event.setdefault("labels", {}).update(self.config.extra_labels)
                yield event

    def _emit(self, events: Iterable[Dict[str, Any]]) -> None:
        for event in events:
            line = json.dumps(event, separators=(",", ":"), sort_keys=True)
            if self._file:
                self._buffer.append(line)
            else:
                print(line)
        if self._file:
            now = time.monotonic()
            if (
                len(self._buffer) >= self.config.flush_every
                or (now - self._last_flush) >= self.config.interval
            ):
                self._flush()

    def _flush(self) -> None:
        assert self._file is not None
        data = "\n".join(self._buffer) + "\n"
        self._file.write(data)
        self._file.flush()
        if self.config.fsync:
            import os

            os.fsync(self._file.fileno())
        self._buffer.clear()
        self._last_flush = time.monotonic()
