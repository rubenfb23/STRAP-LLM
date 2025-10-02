#!/usr/bin/env python3
"""Analiza snapshots generados por tracepoints.py y produce un resumen compacto."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza tracepoints.jsonl y muestra estadísticas por dispositivo.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tracepoints.jsonl"),
        help="Ruta al archivo JSONL exportado por tracepoints.py (default: ./tracepoints.jsonl)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Limita la salida a los N dispositivos con más I/O (0 = todos).",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="50,95,99",
        help="Percentiles de latencia a estimar (separados por coma).",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("benchmarks/systems/I-O"),
        help="Directorio base donde se almacenarán las gráficas generadas.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Deshabilita la generación de gráficas y sólo muestra el resumen por consola.",
    )
    return parser.parse_args()


@dataclass
class DeviceStats:
    name: str = "?"
    total_requests: int = 0
    bucket_counts: Dict[int, int] = field(default_factory=dict)
    queue_samples: int = 0
    queue_sum: int = 0
    queue_max: int = 0
    queue_series: List[Tuple[float, int]] = field(default_factory=list)
    request_series: List[Tuple[float, int]] = field(default_factory=list)

    def register_latency(self, slot: int, count: int) -> None:
        if count <= 0:
            return
        self.bucket_counts[slot] = self.bucket_counts.get(slot, 0) + count
        self.total_requests += count

    def register_queue_depth(self, depth: int) -> None:
        if depth < 0:
            depth = 0
        self.queue_samples += 1
        self.queue_sum += depth
        if depth > self.queue_max:
            self.queue_max = depth

    def latency_mean_us(self) -> Optional[float]:
        if not self.total_requests:
            return None
        acc = 0.0
        for slot, count in self.bucket_counts.items():
            low, high = _log2_bucket_bounds(slot)
            midpoint = (low + high) / 2.0
            acc += midpoint * count
        return acc / self.total_requests

    def latency_percentile_us(self, percentile: float) -> Optional[int]:
        if not self.total_requests:
            return None
        target = self.total_requests * percentile / 100.0
        cumulative = 0
        for slot in sorted(self.bucket_counts):
            cumulative += self.bucket_counts[slot]
            if cumulative >= target:
                _, high = _log2_bucket_bounds(slot)
                return high
        last_slot = max(self.bucket_counts)
        _, last_high = _log2_bucket_bounds(last_slot)
        return last_high

    def avg_queue_depth(self) -> Optional[float]:
        if not self.queue_samples:
            return None
        return self.queue_sum / self.queue_samples


def _log2_bucket_bounds(slot: int) -> Tuple[int, int]:
    low = 0 if slot == 0 else 1 << slot
    high = (1 << (slot + 1)) - 1
    return low, high


def _load_percentiles(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            num = float(chunk)
        except ValueError:
            continue
        if 0 < num < 100:
            values.append(num)
        elif num == 100.0:
            values.append(100.0)
    unique = sorted(set(values))
    return unique


def _read_snapshots(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"Línea {lineno}: JSON inválido") from None


def _prepare_stats() -> Dict[int, DeviceStats]:
    return {}


def _collect_stats(
    snapshot: Dict[str, object], devices: Dict[int, DeviceStats]
) -> None:
    # Ensure we always work with lists of dicts for type-checkers and safety
    raw_latency = snapshot.get("latency_histogram")
    latency_entries: List[Dict[str, object]] = (
        raw_latency if isinstance(raw_latency, list) else []
    )
    raw_inflight = snapshot.get("inflight")
    inflight_entries: List[Dict[str, object]] = (
        raw_inflight if isinstance(raw_inflight, list) else []
    )
    timestamp = float(snapshot.get("timestamp", math.nan))

    devices_in_snapshot = set()

    # Safely build queue_depths with proper int keys/values
    queue_depths: Dict[int, int] = {}
    for entry in inflight_entries:
        if not isinstance(entry, dict):
            continue
        device_id_obj = entry.get("device_id")
        count_obj = entry.get("count", 0)
        try:
            device_id = int(device_id_obj) if device_id_obj is not None else None
        except (TypeError, ValueError):
            device_id = None
        if device_id is None:
            continue
        try:
            count = int(count_obj)
        except (TypeError, ValueError):
            count = 0
        queue_depths[device_id] = count

    for raw in latency_entries:
        if not isinstance(raw, dict):
            continue
        device_id = int(raw.get("device_id", 0))
        name = str(raw.get("device_name", "?"))
        stats = devices.setdefault(device_id, DeviceStats(name=name))
        if not stats.name or stats.name == "?":
            stats.name = name or stats.name
        total_requests = int(raw.get("total", 0))
        if total_requests:
            stats.request_series.append((timestamp, total_requests))
        for bucket in raw.get("buckets", []):
            if not isinstance(bucket, dict):
                continue
            slot = int(bucket.get("slot", 0))
            count = int(bucket.get("count", 0))
            stats.register_latency(slot, count)
        devices_in_snapshot.add(device_id)

    for raw in inflight_entries:
        if not isinstance(raw, dict):
            continue
        device_id = int(raw.get("device_id", 0))
        name = str(raw.get("device_name", "?"))
        stats = devices.setdefault(device_id, DeviceStats(name=name))
        if not stats.name or stats.name == "?":
            stats.name = name or stats.name
        devices_in_snapshot.add(device_id)

    for device_id in devices_in_snapshot:
        depth = queue_depths.get(device_id, 0)
        devices[device_id].register_queue_depth(depth)
        devices[device_id].queue_series.append((timestamp, depth))


def _format_latency_summary(stats: DeviceStats, percentiles: List[float]) -> str:
    parts: List[str] = []
    mean = stats.latency_mean_us()
    if mean is not None:
        parts.append(f"avg≈{mean:.1f}µs")
    for pct in percentiles:
        value = stats.latency_percentile_us(pct)
        if value is not None:
            parts.append(f"p{int(pct)}≤{value}µs")
    if not parts:
        parts.append("sin datos")
    return ", ".join(parts)


def _format_queue_summary(stats: DeviceStats) -> str:
    mean = stats.avg_queue_depth()
    if mean is None:
        return "sin datos"
    return f"avg≈{mean:.2f} | max={stats.queue_max}"


def _slugify(name: str) -> str:
    text = name.strip().lower() or "device"
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "device"


def _generate_charts(
    devices: Dict[int, DeviceStats], charts_dir: Path, percentiles: List[float]
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib no está instalado; omitiendo la generación de gráficas.",
            file=sys.stderr,
        )
        return

    charts_dir.mkdir(parents=True, exist_ok=True)

    for device_id, stats in devices.items():
        slug = _slugify(stats.name)
        base_name = f"{slug}-id{device_id}"

        if stats.bucket_counts:
            slots = sorted(stats.bucket_counts)
            counts = [stats.bucket_counts[slot] for slot in slots]
            labels = [
                f"{_log2_bucket_bounds(slot)[0]}-{_log2_bucket_bounds(slot)[1]}"
                for slot in slots
            ]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(slots)), counts, color="steelblue")
            ax.set_xticks(range(len(slots)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Requests")
            ax.set_xlabel("Latencia (µs, rango por bucket log2)")
            ax.set_title(f"Latencia I/O - {stats.name}")
            ax.grid(True, axis="y", alpha=0.2)
            summary = _format_latency_summary(stats, percentiles)
            ax.text(
                0.99,
                0.95,
                summary,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.6, "boxstyle": "round,pad=0.3"},
                fontsize=8,
            )
            fig.tight_layout()
            fig.savefig(
                charts_dir / f"{base_name}_latency_hist.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        if stats.queue_series:
            times = [pair[0] for pair in stats.queue_series if not math.isnan(pair[0])]
            if times:
                base_ts = min(times)
                series = [
                    (t - base_ts, depth)
                    for t, depth in stats.queue_series
                    if not math.isnan(t)
                ]
                series.sort(key=lambda entry: entry[0])
                rel_times = [entry[0] for entry in series]
                depths = [entry[1] for entry in series]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rel_times, depths, color="darkorange", marker="o", linewidth=1)
                ax.set_xlabel("Tiempo desde el inicio (s)")
                ax.set_ylabel("Requests en vuelo")
                ax.set_title(f"Profundidad de cola - {stats.name}")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(
                    charts_dir / f"{base_name}_queue_depth.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

        if stats.request_series:
            times = [
                pair[0] for pair in stats.request_series if not math.isnan(pair[0])
            ]
            if times:
                base_ts = min(times)
                series = [
                    (t - base_ts, total)
                    for t, total in stats.request_series
                    if not math.isnan(t)
                ]
                series.sort(key=lambda entry: entry[0])
                rel_times = [entry[0] for entry in series]
                totals = [entry[1] for entry in series]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rel_times, totals, color="seagreen", marker="s", linewidth=1)
                ax.set_xlabel("Tiempo desde el inicio (s)")
                ax.set_ylabel("Requests completados por snapshot")
                ax.set_title(f"Throughput por ventana - {stats.name}")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(
                    charts_dir / f"{base_name}_requests.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise SystemExit(f"No se encontró el archivo: {args.input}")

    percentiles = _load_percentiles(args.percentiles)
    devices: Dict[int, DeviceStats] = _prepare_stats()
    snapshots = 0

    for snapshot in _read_snapshots(args.input):
        if not isinstance(snapshot, dict):
            continue
        _collect_stats(snapshot, devices)
        snapshots += 1

    if snapshots == 0:
        print("No se encontraron snapshots en el archivo.")
        return

    sorted_devices = sorted(
        devices.items(),
        key=lambda item: item[1].total_requests,
        reverse=True,
    )
    if args.top > 0:
        sorted_devices = sorted_devices[: args.top]

    print(f"Snapshots procesados: {snapshots}")
    print(f"Dispositivos totales: {len(devices)}")
    print()

    for device_id, stats in sorted_devices:
        print(f"[{stats.name}] (id={device_id})")
        print(f"  requests: {stats.total_requests}")
        print(f"  latencia: {_format_latency_summary(stats, percentiles)}")
        print(f"  cola:     {_format_queue_summary(stats)}")
        print()

    if not args.no_charts:
        _generate_charts(devices, args.charts_dir, percentiles)
        print(f"Gráficas exportadas a: {args.charts_dir}")


if __name__ == "__main__":
    main()
