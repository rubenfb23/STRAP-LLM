"""
Shared helpers for loading BCC programs with resilient search paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .exceptions import BPFUnavailableError

_BCC_FALLBACK_PATHS: tuple[Path, ...] = (
    Path("/usr/share/bcc/python"),
    Path("/usr/lib/python3/dist-packages"),
    Path("/usr/lib/python3/site-packages"),
)


@dataclass(frozen=True)
class BPFHandle:
    """Small wrapper so imports stay lazy until we really need BCC."""

    module: object

    def new(self, *, text: str, cflags: Optional[Iterable[str]] = None) -> "BPF":
        BPF = getattr(self.module, "BPF")
        if cflags is None:
            return BPF(text=text)
        return BPF(text=text, cflags=list(cflags))


def _import_bcc() -> object:
    try:
        from bcc import BPF  # type: ignore[import-untyped]

        return BPF.__module__
    except Exception as exc:  # pragma: no cover - dependent on system availability
        raise BPFUnavailableError(
            "Unable to import bcc.BPF. Install bcc (e.g., `sudo apt-get install bpfcc-tools "
            "python3-bpfcc`) and make sure the python module is on PYTHONPATH."
        ) from exc


def load_bpf_module(extra_paths: Iterable[Path] | None = None) -> BPFHandle:
    """
    Try importing bcc.BPF using well known fallbacks.

    Parameters
    ----------
    extra_paths:
        Additional paths to consider before the built-in fallbacks.
    """

    ordered_paths = list(extra_paths or [])
    ordered_paths.extend(_BCC_FALLBACK_PATHS)

    import importlib
    import sys

    seen = set()
    for candidate in ordered_paths:
        if not candidate.exists():
            continue
        as_str = str(candidate.resolve())
        if as_str in seen:
            continue
        seen.add(as_str)
        if as_str not in sys.path:
            sys.path.insert(0, as_str)

    module = _import_bcc()
    bpf_module = importlib.import_module(module)
    return BPFHandle(module=bpf_module)


def ensure_tracepoints_available(bpf_cls: object, required: Iterable[tuple[str, str]]) -> None:
    missing = [
        f"{category}:{event}"
        for category, event in required
        if not getattr(bpf_cls, "tracepoint_exists")(category, event)
    ]
    if missing:
        raise BPFUnavailableError(
            "Required tracepoints are unavailable: " + ", ".join(missing)
        )
