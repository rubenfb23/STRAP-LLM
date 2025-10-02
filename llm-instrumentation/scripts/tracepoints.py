#!/usr/bin/env python
# Medición de latencia I/O optimizada con tracepoints estables

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

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
            path_obj = Path(candidate)
            if path_obj.exists():
                sys.path.insert(0, str(path_obj))
        from bcc import BPF as _MaybeBPF  # type: ignore[import-not-found]
        BPFClass = _MaybeBPF
        break
    except Exception:
        sys.modules.pop("bcc", None)
        continue

if BPFClass is None:
    raise RuntimeError(
        "No se pudo importar 'bcc.BPF'. Instala bcc (ej.: 'sudo apt-get install bpfcc-tools python3-bpfcc' "
        "y ejecuta como root, o expone la ruta Python de BCC en PYTHONPATH)."
    )

if TYPE_CHECKING:
    from typing import Protocol

    class BPFProtocol(Protocol):  # noqa: D401
        def __init__(self, text: str): ...
        def attach_tracepoint(self, tp: str, fn_name: str) -> None: ...
        @staticmethod
        def tracepoint_exists(category: str, event: str) -> bool: ...
        def get_table(self, name: str) -> Any: ...

else:
    BPFProtocol = Any  # help type checkers when running

from time import sleep

# Validar que los tracepoints existen antes de compilar el programa eBPF. La
# carga del BPF adjunta automáticamente cualquier función cuyo nombre comience
# con "tracepoint__" (macro TRACEPOINT_PROBE), por lo que sólo necesitamos
# asegurarnos de que el ABI está disponible.
_REQUIRED_TRACEPOINTS = (
    ("block", "block_rq_issue"),
    ("block", "block_rq_complete"),
)


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
            + ". Verifica que el kernel exponga block:block_rq_issue y block:block_rq_complete."
        )

# Programa eBPF optimizado basado en tracepoints block_rq_* para ABI estable
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/blk-mq.h>

typedef struct disk_key {
    dev_t dev;
    u64 slot;
} disk_key_t;

struct start_key {
    dev_t dev;
    u32 _pad;
    sector_t sector;
};

BPF_HASH(start, struct start_key);
BPF_HISTOGRAM(latency_hist, disk_key_t);

TRACEPOINT_PROBE(block, block_rq_issue)
{
    struct start_key key = {
        .dev = args->dev,
        .sector = args->sector
    };

    u64 ts = bpf_ktime_get_ns();
    start.update(&key, &ts);
    return 0;
}

TRACEPOINT_PROBE(block, block_rq_complete)
{
    struct start_key key = {
        .dev = args->dev,
        .sector = args->sector
    };

    u64 *tsp = start.lookup(&key);
    if (!tsp) {
        return 0; // solicitud no registrada
    }

    u64 delta = bpf_ktime_get_ns() - *tsp;
    start.delete(&key);

    delta /= 1000; // microsegundos
    if (delta == 0) {
        delta = 1; // evitar log2(0)
    }

    disk_key_t dkey = {};
    dkey.dev = key.dev;
    dkey.slot = bpf_log2l(delta);
    latency_hist.atomic_increment(dkey);

    return 0;
}
"""


def _parse_args() -> argparse.Namespace:
    """Definir flags CLI para muestreo y persistencia."""

    parser = argparse.ArgumentParser(
        description="Mide latencia y cola de I/O usando tracepoints block_rq_* con BCC"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Segundos entre ventanas de muestreo (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tracepoints.jsonl"),
        help="Archivo de salida JSONL para snapshots (default: ./tracepoints.jsonl)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Deshabilita la escritura a disco, deja solo la salida por terminal",
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


class AsyncJSONLWriter:
    """Escritura asíncrona de snapshots JSONL usando ``mmap`` para minimizar copias.

    El uso de memory mapping evita crear buffers intermedios en userland cada vez
    que escribimos al archivo destino y permite crecer el archivo de manera
    amortizada. Esto reduce la presión de memoria/cpu cuando el histograma es
    grande o el intervalo de muestreo es corto.
    """

    def __init__(self, path: Path, flush_every: int, fsync_enabled: bool) -> None:
        self._path = path
        self._flush_every = max(1, flush_every)
        self._fsync_enabled = fsync_enabled
        self._queue: "SimpleQueue[Optional[str]]" = SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="io-trace-writer", daemon=True
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
            raise RuntimeError("Writer no inicializado. Llama start() antes de submit().")
        self._queue.put(line)

    def close(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._thread.join()
        self._started = False

    def _run(self) -> None:
        lines: List[str] = []
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
        except Exception as exc:  # pragma: no cover - proteger el hilo
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

    def _flush(self, state: Dict[str, Any], lines: List[str]) -> None:
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
        except (BufferError, ValueError):  # pragma: no cover - depende del backend mmap
            pass

        if self._fsync_enabled:
            os.fsync(fd)

        lines.clear()

    def _ensure_capacity(self, state: Dict[str, Any], required_length: int) -> mmap.mmap:
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

DISK_LOOKUP: Dict[str, str] = {}


def _refresh_disk_lookup(proc_diskstats: str = "/proc/diskstats") -> None:
    """Actualiza el cache de discos usando ``mmap`` para evitar copias completas."""

    DISK_LOOKUP.clear()
    path = Path(proc_diskstats)
    data = b""
    try:
        with path.open("rb") as stats:
            fd = stats.fileno()
            try:
                with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mm:
                    data = mm[:]
            except (ValueError, BufferError, OSError):
                # Algunos pseudo archivos como /proc no soportan mmap; usamos os.read.
                chunks: List[bytes] = []
                while True:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
    except FileNotFoundError:
        return

    text = data.decode("utf-8", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        major, minor, name = parts[0:3]
        DISK_LOOKUP[f"{major},{minor}"] = name


def disk_print(device_id: int) -> str:
    major = device_id >> 20
    minor = device_id & ((1 << 20) - 1)
    disk = f"{major},{minor}"
    return DISK_LOOKUP.get(disk, "?")


def _log2_bucket_bounds(slot: int) -> Dict[str, int]:
    low = 0 if slot == 0 else 1 << slot
    high = (1 << (slot + 1)) - 1
    return {"low": low, "high": high}


def _collect_latency_snapshot(table: Any) -> List[Dict[str, Any]]:
    per_device: Dict[int, List[Dict[str, int]]] = {}
    for key, leaf in table.items():
        dev = int(getattr(key, "dev", 0))
        slot = int(getattr(key, "slot", 0))
        count = int(getattr(leaf, "value", 0))
        bounds = _log2_bucket_bounds(slot)
        per_device.setdefault(dev, []).append(
            {
                "slot": slot,
                "count": count,
                "bucket_low": bounds["low"],
                "bucket_high": bounds["high"],
            }
        )

    snapshot: List[Dict[str, Any]] = []
    for dev, buckets in per_device.items():
        buckets.sort(key=lambda entry: entry["slot"])
        snapshot.append(
            {
                "device_id": dev,
                "device_name": disk_print(dev),
                "total": sum(bucket["count"] for bucket in buckets),
                "buckets": buckets,
            }
        )
    snapshot.sort(key=lambda entry: entry["device_name"])
    return snapshot


def _collect_inflight_snapshot(table: Any) -> List[Dict[str, Any]]:
    counts: Dict[int, int] = {}
    for key, _ in table.items():
        dev = int(getattr(key, "dev", 0))
        counts[dev] = counts.get(dev, 0) + 1

    inflight = [
        {"device_id": dev, "device_name": disk_print(dev), "count": depth}
        for dev, depth in counts.items()
    ]
    inflight.sort(key=lambda entry: entry["count"], reverse=True)
    return inflight


def _snapshot_to_json(
    timestamp: float,
    interval_s: float,
    latency: Iterable[Dict[str, Any]],
    inflight: Iterable[Dict[str, Any]],
) -> str:
    iso_ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "iso_timestamp": iso_ts,
        "interval_s": interval_s,
        "latency_histogram": list(latency),
        "inflight": list(inflight),
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def main() -> None:
    args = _parse_args()

    if args.interval <= 0:
        sys.exit("--interval debe ser mayor que 0")

    if os.geteuid() != 0:
        sys.exit(
            "Este script necesita ejecutarse como root (ej. 'sudo python3 llm-instrumentation/scripts/tracepoints.py') "
            "para poder inicializar los tracepoints block:block_rq_* con BCC."
        )

    writer: Optional[AsyncJSONLWriter] = None
    if not args.no_output:
        writer = AsyncJSONLWriter(args.output, args.flush_every, args.fsync)
        writer.start()

    _validate_tracepoints()

    b: BPFProtocol = BPFClass(text=bpf_text)

    _refresh_disk_lookup()

    dist = b.get_table("latency_hist")
    start_map = b.get_table("start")

    try:
        while True:
            sleep(args.interval)
            snapshot_ts = time.time()
            if writer is not None:
                latency_snapshot = _collect_latency_snapshot(dist)
                inflight_snapshot = _collect_inflight_snapshot(start_map)
                writer.submit(
                    _snapshot_to_json(
                        snapshot_ts, args.interval, latency_snapshot, inflight_snapshot
                    )
                )
            dist.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
