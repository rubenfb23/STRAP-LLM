#!/usr/bin/env python
# Medición de latencia I/O optimizada con tracepoints estables

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import argparse
import json
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

from time import sleep, strftime

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
    """Escritura asíncrona de snapshots JSONL para minimizar overhead."""

    def __init__(self, path: Path, flush_every: int, fsync_enabled: bool) -> None:
        self._path = path
        self._flush_every = max(1, flush_every)
        self._fsync_enabled = fsync_enabled
        self._queue: "SimpleQueue[Optional[str]]" = SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="io-trace-writer", daemon=True
        )
        self._started = False

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
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    lines.append(item)
                    if len(lines) >= self._flush_every:
                        self._flush(handle, lines)
                if lines:
                    self._flush(handle, lines)
        except Exception as exc:  # pragma: no cover - proteger el hilo
            print(f"[tracepoints] Error en escritor asíncrono: {exc}", file=sys.stderr)

    def _flush(self, handle: Any, lines: List[str]) -> None:
        payload = "\n".join(lines) + "\n"
        handle.write(payload)
        handle.flush()
        if self._fsync_enabled:
            os.fsync(handle.fileno())
        lines.clear()

DISK_LOOKUP: Dict[str, str] = {}


def _refresh_disk_lookup(proc_diskstats: str = "/proc/diskstats") -> None:
    DISK_LOOKUP.clear()
    with open(proc_diskstats) as stats:
        for line in stats:
            major, minor, name = line.split()[0:3]
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

    print(
        "Trazando latencia I/O... Presiona Ctrl-C para terminar (intervalo = "
        f"{args.interval:.1f}s)"
    )

    try:
        while True:
            sleep(args.interval)
            snapshot_ts = time.time()

            latency_snapshot = _collect_latency_snapshot(dist)

            print(f"\n{strftime('%H:%M:%S')} - Latencia I/O por dispositivo:")
            dist.print_log2_hist("microsegundos", "disk", disk_print)
            dist.clear()

            inflight_snapshot = _collect_inflight_snapshot(start_map)

            if inflight_snapshot:
                print("Cola I/O en vuelo (requests):")
                for entry in inflight_snapshot:
                    print(f"  {entry['device_name']:>12} -> {entry['count']}")
            else:
                print("Cola I/O en vuelo (requests): sin pendientes")

            if writer is not None:
                writer.submit(
                    _snapshot_to_json(
                        snapshot_ts, args.interval, latency_snapshot, inflight_snapshot
                    )
                )
    except KeyboardInterrupt:
        print("\nFinalizando...")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
