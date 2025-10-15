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
from typing import Dict, Iterable, List, Optional, Set, Tuple


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
    parser.add_argument(
        "--chart-types",
        type=str,
        default="all",
        help=(
            "Tipos de gráficas a generar (lista separada por comas). "
            "Opciones: hist,queue,requests,percentiles,heatmap,qdepth_iops. "
            "Usa 'all' para habilitar todas."
        ),
    )
    return parser.parse_args()


@dataclass
class LatencySnapshot:
    timestamp: float
    interval_s: float
    total_requests: int
    bucket_counts: Dict[int, int]


@dataclass
class DeviceStats:
    name: str = "?"
    total_requests: int = 0
    bucket_counts: Dict[int, int] = field(default_factory=dict)
    queue_samples: int = 0
    queue_sum: int = 0
    queue_max: int = 0
    queue_series: List[Tuple[float, int]] = field(default_factory=list)
    request_series: List[Tuple[float, int, float]] = field(default_factory=list)
    latency_snapshots: List[LatencySnapshot] = field(default_factory=list)

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
        return _compute_percentiles(self.bucket_counts, [percentile]).get(percentile)

    def avg_queue_depth(self) -> Optional[float]:
        if not self.queue_samples:
            return None
        return self.queue_sum / self.queue_samples


DEFAULT_CHART_TYPES: Set[str] = {
    "hist",
    "queue",
    "requests",
    "percentiles",
    "heatmap",
    "qdepth_iops",
}


def _log2_bucket_bounds(slot: int) -> Tuple[int, int]:
    low = 0 if slot == 0 else 1 << slot
    high = (1 << (slot + 1)) - 1
    return low, high


def _compute_percentiles(
    bucket_counts: Dict[int, int], percentiles: Iterable[float]
) -> Dict[float, Optional[int]]:
    total = sum(bucket_counts.values())
    results: Dict[float, Optional[int]] = {}
    if total <= 0:
        for pct in percentiles:
            results[pct] = None
        return results

    sorted_slots = sorted(bucket_counts)
    for pct in percentiles:
        target = total * pct / 100.0
        cumulative = 0
        resolved: Optional[int] = None
        for slot in sorted_slots:
            cumulative += bucket_counts[slot]
            if cumulative >= target:
                _, high = _log2_bucket_bounds(slot)
                resolved = high
                break
        if resolved is None and sorted_slots:
            last_slot = sorted_slots[-1]
            _, high = _log2_bucket_bounds(last_slot)
            resolved = high
        results[pct] = resolved
    return results


def _prepare_latency_time_data(
    stats: DeviceStats, percentiles: List[float]
) -> Optional[Dict[str, object]]:
    valid_snapshots = [
        snap for snap in stats.latency_snapshots if not math.isnan(snap.timestamp)
    ]
    if not valid_snapshots:
        return None

    valid_snapshots.sort(key=lambda snap: snap.timestamp)
    base_ts = valid_snapshots[0].timestamp
    rel_times = [snap.timestamp - base_ts for snap in valid_snapshots]

    bucket_order = sorted(
        {slot for snap in valid_snapshots for slot in snap.bucket_counts}
    )
    if not bucket_order:
        bucket_order = sorted(stats.bucket_counts)

    percentile_series: Dict[float, List[float]] = {
        pct: [] for pct in percentiles
    }
    heatmap_rows: List[List[float]] = []

    for snap in valid_snapshots:
        pct_values = _compute_percentiles(snap.bucket_counts, percentiles)
        for pct in percentiles:
            value = pct_values.get(pct)
            percentile_series[pct].append(float(value) if value is not None else math.nan)

        total = snap.total_requests
        if total <= 0:
            total = sum(snap.bucket_counts.values())
        if total <= 0 or not bucket_order:
            heatmap_rows.append([0.0 for _ in bucket_order])
        else:
            heatmap_rows.append(
                [snap.bucket_counts.get(slot, 0) / total for slot in bucket_order]
            )

    return {
        "base_ts": base_ts,
        "rel_times": rel_times,
        "percentile_series": percentile_series,
        "bucket_order": bucket_order,
        "heatmap_rows": heatmap_rows,
    }


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


def _parse_chart_types(raw: str) -> Set[str]:
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not tokens or "all" in tokens:
        return set(DEFAULT_CHART_TYPES)
    resolved = {token for token in tokens if token in DEFAULT_CHART_TYPES}
    if not resolved:
        return set(DEFAULT_CHART_TYPES)
    return resolved


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
    interval_s = float(snapshot.get("interval_s", math.nan))

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
        if not math.isnan(timestamp):
            stats.request_series.append((timestamp, total_requests, interval_s))
        snapshot_buckets: Dict[int, int] = {}
        for bucket in raw.get("buckets", []):
            if not isinstance(bucket, dict):
                continue
            slot = int(bucket.get("slot", 0))
            count = int(bucket.get("count", 0))
            stats.register_latency(slot, count)
            snapshot_buckets[slot] = snapshot_buckets.get(slot, 0) + count
        if not math.isnan(timestamp):
            stats.latency_snapshots.append(
                LatencySnapshot(
                    timestamp=timestamp,
                    interval_s=interval_s,
                    total_requests=total_requests,
                    bucket_counts=snapshot_buckets,
                )
            )
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
        if not math.isnan(timestamp):
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
    devices: Dict[int, DeviceStats],
    charts_dir: Path,
    percentiles: List[float],
    enabled_charts: Set[str],
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
        time_data = (
            _prepare_latency_time_data(stats, percentiles)
            if enabled_charts.intersection({"percentiles", "heatmap"})
            else None
        )

        if "hist" in enabled_charts and stats.bucket_counts:
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

        if time_data:
            rel_times = time_data["rel_times"]
            percentile_series = time_data["percentile_series"]
            bucket_order = time_data["bucket_order"]
            heatmap_rows = time_data["heatmap_rows"]

            if (
                "percentiles" in enabled_charts
                and rel_times
                and percentiles
            ):
                fig, ax = plt.subplots(figsize=(10, 4))
                plotted_lines = []
                for pct in percentiles:
                    values = percentile_series.get(pct, [])
                    if not values or all(math.isnan(value) for value in values):
                        continue
                    line, = ax.plot(
                        rel_times,
                        values,
                        linewidth=1.5,
                        marker="o",
                        label=f"p{pct:g}",
                    )
                    plotted_lines.append(line)
                if plotted_lines:
                    ax.set_xlabel("Tiempo desde el inicio (s)")
                    ax.set_ylabel("Latencia (µs)")
                    ax.set_title(f"Percentiles de latencia - {stats.name}")
                    ax.grid(True, alpha=0.2)
                    ax.legend(loc="upper right")
                    fig.tight_layout()
                    fig.savefig(
                        charts_dir / f"{base_name}_latency_percentiles.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                plt.close(fig)

            if (
                "heatmap" in enabled_charts
                and rel_times
                and bucket_order
                and heatmap_rows
                and any(sum(row) > 0 for row in heatmap_rows)
            ):
                heatmap_matrix = [
                    list(values) for values in zip(*heatmap_rows)
                ]  # buckets x snapshots
                if heatmap_matrix:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if len(rel_times) > 1:
                        avg_step = (rel_times[-1] - rel_times[0]) / (
                            len(rel_times) - 1
                        )
                        if avg_step <= 0:
                            avg_step = 1.0
                    else:
                        avg_step = 1.0
                    x_min = rel_times[0]
                    x_max = rel_times[-1] + avg_step
                    im = ax.imshow(
                        heatmap_matrix,
                        aspect="auto",
                        origin="lower",
                        interpolation="nearest",
                        extent=[x_min, x_max, -0.5, len(bucket_order) - 0.5],
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    y_ticks = list(range(len(bucket_order)))
                    y_labels = [
                        f"{_log2_bucket_bounds(slot)[0]}-{_log2_bucket_bounds(slot)[1]}"
                        for slot in bucket_order
                    ]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    ax.set_xlabel("Tiempo desde el inicio (s)")
                    ax.set_ylabel("Latencia (µs, bucket log2)")
                    ax.set_title(f"Distribución de latencias - {stats.name}")
                    cbar = fig.colorbar(im, ax=ax, pad=0.01)
                    cbar.set_label("Fracción de requests")
                    fig.tight_layout()
                    fig.savefig(
                        charts_dir / f"{base_name}_latency_heatmap.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

        if "queue" in enabled_charts and stats.queue_series:
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

        if "requests" in enabled_charts and stats.request_series:
            times = [
                pair[0] for pair in stats.request_series if not math.isnan(pair[0])
            ]
            if times:
                base_ts = min(times)
                series = [
                    (t - base_ts, total)
                    for t, total, _interval in stats.request_series
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

        queue_points = [
            (t, depth) for t, depth in stats.queue_series if not math.isnan(t)
        ]
        throughput_points = [
            (t, total, interval)
            for t, total, interval in stats.request_series
            if not math.isnan(t)
        ]
        if "qdepth_iops" in enabled_charts and (queue_points or throughput_points):
            time_candidates = [t for t, _ in queue_points] + [
                t for t, _, _ in throughput_points
            ]
            if time_candidates:
                base_ts = min(time_candidates)
                fig, ax_queue = plt.subplots(figsize=(10, 4))
                legend_handles = []
                legend_labels = []

                if queue_points:
                    queue_points.sort(key=lambda entry: entry[0])
                    rel_times_queue = [t - base_ts for t, _ in queue_points]
                    queue_values = [depth for _, depth in queue_points]
                    queue_line, = ax_queue.plot(
                        rel_times_queue,
                        queue_values,
                        color="darkorange",
                        marker="o",
                        linewidth=1,
                        label="Cola",
                    )
                    legend_handles.append(queue_line)
                    legend_labels.append("Cola")

                ax_queue.set_xlabel("Tiempo desde el inicio (s)")
                ax_queue.set_ylabel("Requests en vuelo")
                ax_queue.grid(True, alpha=0.2)

                if throughput_points:
                    throughput_points.sort(key=lambda entry: entry[0])
                    rel_times_iops = [t - base_ts for t, _, _ in throughput_points]
                    throughput_values: List[float] = []
                    for _, total, interval in throughput_points:
                        if math.isnan(interval) or interval <= 0:
                            throughput_values.append(float(total))
                        else:
                            throughput_values.append(total / interval)
                    ax_iops = ax_queue.twinx()
                    iops_line, = ax_iops.plot(
                        rel_times_iops,
                        throughput_values,
                        color="seagreen",
                        marker="s",
                        linewidth=1,
                        label="Requests/s",
                    )
                    ax_iops.set_ylabel("Requests completados por segundo")
                    legend_handles.append(iops_line)
                    legend_labels.append("Requests/s")

                if legend_handles:
                    ax_queue.legend(legend_handles, legend_labels, loc="upper right")
                ax_queue.set_title(f"Cola vs throughput - {stats.name}")
                fig.tight_layout()
                fig.savefig(
                    charts_dir / f"{base_name}_qdepth_iops.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        args.input.parent.mkdir(parents=True, exist_ok=True)
        args.input.touch(exist_ok=True)
        print(
            f"Se creó el archivo vacío: {args.input}. "
            "Ejecuta tracepoints.py para recolectar snapshots."
        )
        return

    percentiles = _load_percentiles(args.percentiles)
    chart_types = _parse_chart_types(args.chart_types)
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
        _generate_charts(devices, args.charts_dir, percentiles, chart_types)
        print(f"Gráficas exportadas a: {args.charts_dir}")


if __name__ == "__main__":
    main()
