#!/usr/bin/env python3
# Monitor de fallos de página, tiempo fuera de CPU y presión PSI usando BCC

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import argparse
import json
import mmap
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue

# Basado en la guía oficial de BCC: https://github.com/iovisor/bcc/blob/master/docs/tutorial_bcc_python_developer.md
_BCC_FALLBACK_PATHS = (
    "/usr/share/bcc/python",
    "/usr/lib/python3/dist-packages",
    "/usr/lib/python3/site-packages",
)

BPFClass = None
for candidate in (None,) + _BCC_FALLBACK_PATHS:
    try:
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                sys.path.insert(0, str(candidate_path))
        from bcc import BPF as _MaybeBPF  # type: ignore[import-not-found]

        BPFClass = _MaybeBPF
        break
    except Exception:
        sys.modules.pop("bcc", None)
        continue

if BPFClass is None:
    raise RuntimeError(
        "No se pudo importar 'bcc.BPF'. Instala bcc (ej.: 'sudo apt-get install bpfcc-tools python3-bpfcc'), "
        "asegúrate de ejecutarlo como root o expone la ruta Python de BCC en PYTHONPATH."
    )

if TYPE_CHECKING:
    from typing import Protocol

    class BPFProtocol(Protocol):
        def __init__(self, text: str): ...
        def attach_tracepoint(self, tp: str, fn_name: str) -> None: ...
        @staticmethod
        def tracepoint_exists(category: str, event: str) -> bool: ...
        def get_table(self, name: str) -> Any: ...

else:
    BPFProtocol = Any

_REQUIRED_TRACEPOINTS = (
    ("exceptions", "page_fault_user"),
    ("sched", "sched_switch"),
)

bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

BPF_HASH(page_faults, u32, u64);
BPF_HASH(offcpu_ns, u32, u64);
BPF_HASH(offcpu_start, u32, u64);

TRACEPOINT_PROBE(exceptions, page_fault_user)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    if (pid == 0) {
        return 0;
    }

    u64 one = 1;
    u64 *existing = page_faults.lookup(&pid);
    if (existing) {
        __sync_fetch_and_add(existing, 1);
    } else {
        page_faults.update(&pid, &one);
    }
    return 0;
}

TRACEPOINT_PROBE(sched, sched_switch)
{
    u64 ts = bpf_ktime_get_ns();
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;

    if (prev_pid > 0) {
        offcpu_start.update(&prev_pid, &ts);
    }

    if (next_pid > 0) {
        u64 *start_ts = offcpu_start.lookup(&next_pid);
        if (start_ts) {
            u64 delta = ts - *start_ts;
            offcpu_start.delete(&next_pid);

            u64 *total = offcpu_ns.lookup(&next_pid);
            if (total) {
                delta += *total;
            }
            offcpu_ns.update(&next_pid, &delta);
        }
    }

    return 0;
}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recolecta métricas de CPU/memoria: fallos de página, tiempo off-CPU y presión PSI"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Segundos entre snapshots de métricas (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("system_metrics.jsonl"),
        help="Archivo JSONL de salida para snapshots (default: ./system_metrics.jsonl)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Deshabilita escritura a disco (las muestras se imprimen por stdout)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=12,
        help="Cantidad de snapshots antes de forzar flush en disco (default: 12)",
    )
    parser.add_argument(
        "--fsync",
        action="store_true",
        help="Forzar fsync tras cada flush (mayor seguridad, más overhead)",
    )
    return parser.parse_args()


def _validate_tracepoints() -> None:
    missing = [
        f"{category}:{event}"
        for category, event in _REQUIRED_TRACEPOINTS
        if not BPFClass.tracepoint_exists(category, event)
    ]
    if missing:
        sys.exit(
            "Tracepoints requeridos no disponibles: "
            + ", ".join(missing)
            + ". Verifica que el kernel expone mm:page_fault_user y sched:sched_switch."
        )


class AsyncJSONLWriter:
    """Escritor JSONL asíncrono inspirado en tracepoints.py para minimizar overhead."""

    def __init__(self, path: Path, flush_every: int, fsync_enabled: bool) -> None:
        self._path = path
        self._flush_every = max(1, flush_every)
        self._fsync_enabled = fsync_enabled
        self._queue: "SimpleQueue[Optional[str]]" = SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="system-metrics-writer", daemon=True
        )
        self._started = False
        self._min_map_size = max(1, mmap.ALLOCATIONGRANULARITY)
        self._last_error: Optional[Exception] = None

    def start(self) -> None:
        if self._started:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        self._started = True

    def submit(self, line: str) -> None:
        if not self._started:
            raise RuntimeError(
                "Writer no inicializado. Llama start() antes de submit()."
            )
        self._queue.put(line)

    def close(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join()
        self._started = False

    def _run(self) -> None:
        lines: list[str] = []
        state: Dict[str, Any] = {
            "fd": None,
            "mm": None,
            "map_length": 0,
            "write_offset": 0,
            "data_length": 0,
        }
        try:
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT)
            state["fd"] = fd

            existing_size = os.fstat(fd).st_size
            state["data_length"] = existing_size
            state["write_offset"] = existing_size

            map_length = max(existing_size, self._min_map_size)
            if map_length == 0:
                map_length = self._min_map_size
            if existing_size < map_length:
                os.ftruncate(fd, map_length)

            state["mm"] = mmap.mmap(fd, max(map_length, 1), access=mmap.ACCESS_WRITE)
            state["map_length"] = max(map_length, 1)

            while True:
                item = self._queue.get()
                if item is None:
                    break
                lines.append(item)
                if len(lines) >= self._flush_every:
                    self._flush(state, lines)
            if lines:
                self._flush(state, lines)
        except Exception as exc:
            self._last_error = exc
        finally:
            mm_obj = state.get("mm")
            fd_obj = state.get("fd")
            data_length = int(state.get("data_length") or 0)
            if mm_obj is not None:
                try:
                    mm_obj.flush()
                finally:
                    mm_obj.close()
            if fd_obj is not None:
                try:
                    os.ftruncate(fd_obj, data_length)
                except OSError:
                    pass
                os.close(fd_obj)

    def _flush(self, state: Dict[str, Any], lines: list[str]) -> None:
        mm_obj: Optional[mmap.mmap] = state.get("mm")
        fd = state.get("fd")
        if mm_obj is None or fd is None:
            return

        payload = "\n".join(lines) + "\n"
        encoded = payload.encode("utf-8")

        required_length = state["write_offset"] + len(encoded)
        mm_obj = self._ensure_capacity(state, required_length)

        start = state["write_offset"]
        mm_obj[start : start + len(encoded)] = encoded
        state["write_offset"] = start + len(encoded)
        state["data_length"] = state["write_offset"]

        try:
            mm_obj.flush()
        except (BufferError, ValueError):
            pass

        if self._fsync_enabled:
            os.fsync(fd)

        lines.clear()

    def _ensure_capacity(
        self, state: Dict[str, Any], required_length: int
    ) -> mmap.mmap:
        mm_obj: mmap.mmap = state["mm"]
        fd: int = state["fd"]
        current_length = max(int(state.get("map_length", 0)), 1)

        if required_length <= current_length:
            return mm_obj

        new_length = current_length
        while new_length < required_length:
            new_length *= 2

        mm_obj.flush()
        mm_obj.close()

        os.ftruncate(fd, new_length)
        state["mm"] = mmap.mmap(fd, new_length, access=mmap.ACCESS_WRITE)
        state["map_length"] = new_length
        return state["mm"]


def _collect_pid_map(table: Any) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for key, value in table.items():
        pid = int(getattr(key, "value", getattr(key, "pid", 0)))
        metric = int(getattr(value, "value", 0))
        if pid <= 0:
            continue
        result[str(pid)] = metric
    table.clear()
    return result


def _parse_pressure_line(line: str) -> tuple[str, Dict[str, float]]:
    tokens = line.strip().split()
    if not tokens:
        return "", {}
    scope = tokens[0]
    metrics: Dict[str, float] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            if "." in value:
                metrics[key] = float(value)
            else:
                metrics[key] = float(int(value))
        except ValueError:
            continue
    return scope, metrics


def _read_pressure(resource: str) -> Dict[str, Dict[str, float]]:
    path = Path("/proc/pressure") / resource
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    payload: Dict[str, Dict[str, float]] = {}
    for line in text.splitlines():
        scope, metrics = _parse_pressure_line(line)
        if scope:
            payload[scope] = metrics
    return payload


def _snapshot_to_json(
    timestamp: float,
    interval_s: float,
    offcpu_ns: Dict[str, int],
    page_faults: Dict[str, int],
    pressure: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    iso_ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "iso_timestamp": iso_ts,
        "interval_s": interval_s,
        "off_cpu_ns": offcpu_ns,
        "page_faults": page_faults,
        "pressure": pressure,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _collect_pressure_snapshot() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        "cpu": _read_pressure("cpu"),
        "io": _read_pressure("io"),
        "memory": _read_pressure("memory"),
    }


def main() -> None:
    args = _parse_args()

    if args.interval <= 0:
        sys.exit("--interval debe ser mayor que 0")

    if os.geteuid() != 0:
        sys.exit(
            "Este script necesita ejecutarse como root (ej. 'sudo python3 llm-instrumentation/scripts/system_metrics.py')."
        )

    writer: Optional[AsyncJSONLWriter] = None
    if not args.no_output:
        writer = AsyncJSONLWriter(args.output, args.flush_every, args.fsync)
        writer.start()

    _validate_tracepoints()

    bpf: BPFProtocol = BPFClass(text=bpf_text)
    offcpu_table = bpf.get_table("offcpu_ns")
    page_fault_table = bpf.get_table("page_faults")

    try:
        while True:
            time.sleep(args.interval)
            snapshot_ts = time.time()

            offcpu_snapshot = _collect_pid_map(offcpu_table)
            page_fault_snapshot = _collect_pid_map(page_fault_table)
            pressure_snapshot = _collect_pressure_snapshot()

            line = _snapshot_to_json(
                snapshot_ts,
                args.interval,
                offcpu_snapshot,
                page_fault_snapshot,
                pressure_snapshot,
            )

            if writer is not None:
                writer.submit(line)
            if args.no_output:
                print(line)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
