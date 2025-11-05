"""
Helpers to propagate correlation IDs from application code into collector output.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

DEFAULT_KEY = contextvars.ContextVar("llm_prof_corr_id", default=None)


@dataclass
class CorrelationContext:
    enabled: bool
    _var: contextvars.ContextVar[Optional[str]]

    @classmethod
    def default(cls) -> "CorrelationContext":
        return cls(enabled=True, _var=DEFAULT_KEY)

    @classmethod
    def disabled(cls) -> "CorrelationContext":
        return cls(enabled=False, _var=DEFAULT_KEY)

    def current_id(self) -> Optional[str]:
        if not self.enabled:
            return None
        return self._var.get()

    def set(self, corr_id: Optional[str]) -> None:
        if not self.enabled:
            return
        self._var.set(corr_id)

    def new_request(self) -> str:
        if not self.enabled:
            raise RuntimeError("Correlation context disabled")
        corr_id = f"{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
        self.set(corr_id)
        return corr_id


__all__ = ["CorrelationContext", "DEFAULT_KEY"]
