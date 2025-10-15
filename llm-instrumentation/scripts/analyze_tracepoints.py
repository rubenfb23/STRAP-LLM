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
        "--system-metrics",
        type=Path,
        default=Path("system_metrics.jsonl"),
        help="Ruta al archivo JSONL exportado por system_metrics.py (default: ./system_metrics.jsonl)",
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
            "Opciones: hist,queue,requests,percentiles,heatmap,qdepth_iops,importance. "
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


@dataclass
class SystemSnapshot:
    timestamp: float
    interval_s: float
    off_cpu_ns: Dict[str, int]
    page_faults: Dict[str, int]
    pressure: Dict[str, Dict[str, Dict[str, float]]]


@dataclass
class SystemMetricsData:
    snapshots: List[SystemSnapshot] = field(default_factory=list)
    total_offcpu_ns: Dict[str, float] = field(default_factory=dict)
    total_page_faults: Dict[str, int] = field(default_factory=dict)


DEFAULT_CHART_TYPES: Set[str] = {
    "hist",
    "queue",
    "requests",
    "percentiles",
    "heatmap",
    "qdepth_iops",
    "importance",
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
                print(
                    f"Aviso: línea {lineno} en {path} no contiene JSON válido, se omite.",
                    file=sys.stderr,
                )
                continue


def _parse_pid_map(raw: object) -> Dict[str, int]:
    result: Dict[str, int] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            pid = str(key)
            try:
                metric = int(value)
            except (TypeError, ValueError):
                continue
            result[pid] = metric
    return result


def _sanitize_pressure(raw: object) -> Dict[str, Dict[str, Dict[str, float]]]:
    cleaned: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not isinstance(raw, dict):
        return cleaned
    for resource, scopes in raw.items():
        if not isinstance(scopes, dict):
            continue
        resource_payload: Dict[str, Dict[str, float]] = {}
        for scope, metrics in scopes.items():
            if not isinstance(metrics, dict):
                continue
            metric_payload: Dict[str, float] = {}
            for key, value in metrics.items():
                try:
                    metric_payload[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if metric_payload:
                resource_payload[scope] = metric_payload
        if resource_payload:
            cleaned[str(resource)] = resource_payload
    return cleaned


def _read_system_metrics(path: Path) -> Optional[SystemMetricsData]:
    if not path.exists():
        return None

    data = SystemMetricsData()
    for snapshot in _read_snapshots(path):
        if not isinstance(snapshot, dict):
            continue
        timestamp = float(snapshot.get("timestamp", math.nan))
        if math.isnan(timestamp):
            continue
        interval_s = float(snapshot.get("interval_s", math.nan))

        off_cpu_map = _parse_pid_map(snapshot.get("off_cpu_ns"))
        page_faults_map = _parse_pid_map(snapshot.get("page_faults"))
        pressure = _sanitize_pressure(snapshot.get("pressure"))

        data.snapshots.append(
            SystemSnapshot(
                timestamp=timestamp,
                interval_s=interval_s,
                off_cpu_ns=off_cpu_map,
                page_faults=page_faults_map,
                pressure=pressure,
            )
        )

        for pid, value in off_cpu_map.items():
            data.total_offcpu_ns[pid] = data.total_offcpu_ns.get(pid, 0.0) + float(
                value
            )
        for pid, value in page_faults_map.items():
            data.total_page_faults[pid] = data.total_page_faults.get(pid, 0) + int(
                value
            )

    if not data.snapshots:
        return None

    data.snapshots.sort(key=lambda snap: snap.timestamp)
    return data


def _safe_interval(value: float) -> float:
    if math.isnan(value) or value <= 0:
        return 1.0
    return value


def _extract_pressure_value(
    pressure: Dict[str, Dict[str, Dict[str, float]]], resource: str
) -> Optional[float]:
    resource_payload = pressure.get(resource)
    if not resource_payload:
        return None
    some_scope = resource_payload.get("some")
    if not some_scope:
        # Tomar cualquier scope disponible (ej. "full")
        for metrics in resource_payload.values():
            if metrics:
                some_scope = metrics
                break
    if not some_scope:
        return None
    for key in ("avg10", "avg60", "avg300", "total", "stall"):
        value = some_scope.get(key)
        if value is not None:
            return float(value)
    # Como fallback, devolver el primer valor disponible
    if some_scope:
        return float(next(iter(some_scope.values())))
    return None


def _align_snapshots(
    latency_snapshots: List[LatencySnapshot],
    system_snapshots: List[SystemSnapshot],
) -> List[Tuple[LatencySnapshot, SystemSnapshot]]:
    if not latency_snapshots or not system_snapshots:
        return []
    latency_sorted = sorted(latency_snapshots, key=lambda snap: snap.timestamp)
    system_sorted = sorted(system_snapshots, key=lambda snap: snap.timestamp)

    aligned: List[Tuple[LatencySnapshot, SystemSnapshot]] = []
    i = 0
    j = 0
    while i < len(latency_sorted) and j < len(system_sorted):
        lat_snap = latency_sorted[i]
        sys_snap = system_sorted[j]
        tolerance = max(
            _safe_interval(lat_snap.interval_s),
            _safe_interval(sys_snap.interval_s),
            1.0,
        ) * 0.75
        delta = sys_snap.timestamp - lat_snap.timestamp
        if abs(delta) <= tolerance:
            aligned.append((lat_snap, sys_snap))
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return aligned


def _pearson_corr(pairs: List[Tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for x, y in pairs:
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    if denom_x <= 0 or denom_y <= 0:
        return None
    return num / math.sqrt(denom_x * denom_y)


def _compute_importance_payload(
    latency_snapshots: List[LatencySnapshot],
    system_data: Optional[SystemMetricsData],
    percentiles: List[float],
) -> Optional[Dict[str, object]]:
    if not system_data or not latency_snapshots:
        return None

    aligned = _align_snapshots(latency_snapshots, system_data.snapshots)
    if not aligned:
        top_offcpu = sorted(
            (
                (pid, total_ns / 1_000_000.0)
                for pid, total_ns in system_data.total_offcpu_ns.items()
                if total_ns > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        top_pagefaults = sorted(
            (
                (pid, count)
                for pid, count in system_data.total_page_faults.items()
                if count > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        if not top_offcpu and not top_pagefaults:
            return None
        return {
            "correlations": [],
            "time_series": [],
            "top_offcpu": top_offcpu,
            "top_pagefaults": top_pagefaults,
            "aligned_pairs": 0,
        }

    target_percentile = max(percentiles) if percentiles else 95.0
    feature_points: Dict[str, List[Tuple[float, float]]] = {
        "off_cpu_ms": [],
        "page_faults_per_s": [],
        "cpu_pressure": [],
        "io_pressure": [],
        "memory_pressure": [],
    }
    time_series: List[Dict[str, float]] = []
    base_ts = aligned[0][0].timestamp

    for lat_snap, sys_snap in aligned:
        percentile_value = _compute_percentiles(
            lat_snap.bucket_counts, [target_percentile]
        ).get(target_percentile)
        if percentile_value is None:
            continue
        total_offcpu_ns = sum(sys_snap.off_cpu_ns.values())
        total_page_faults = sum(sys_snap.page_faults.values())
        interval = _safe_interval(
            sys_snap.interval_s if not math.isnan(sys_snap.interval_s) else lat_snap.interval_s
        )
        off_cpu_ms = total_offcpu_ns / 1_000_000.0
        page_faults_per_s = (
            total_page_faults / interval if interval > 0 else float(total_page_faults)
        )
        cpu_pressure = _extract_pressure_value(sys_snap.pressure, "cpu")
        io_pressure = _extract_pressure_value(sys_snap.pressure, "io")
        mem_pressure = _extract_pressure_value(sys_snap.pressure, "memory")

        feature_points["off_cpu_ms"].append((percentile_value, off_cpu_ms))
        feature_points["page_faults_per_s"].append(
            (percentile_value, page_faults_per_s)
        )
        if cpu_pressure is not None:
            feature_points["cpu_pressure"].append((percentile_value, cpu_pressure))
        if io_pressure is not None:
            feature_points["io_pressure"].append((percentile_value, io_pressure))
        if mem_pressure is not None:
            feature_points["memory_pressure"].append((percentile_value, mem_pressure))

        time_series.append(
            {
                "time": lat_snap.timestamp - base_ts,
                "latency_us": float(percentile_value),
                "off_cpu_ms": off_cpu_ms,
                "page_faults_per_s": page_faults_per_s,
                "cpu_pressure": float(cpu_pressure)
                if cpu_pressure is not None
                else math.nan,
                "io_pressure": float(io_pressure)
                if io_pressure is not None
                else math.nan,
                "memory_pressure": float(mem_pressure)
                if mem_pressure is not None
                else math.nan,
            }
        )

    correlations: List[Tuple[str, float]] = []
    for key, pairs in feature_points.items():
        corr = _pearson_corr(pairs)
        if corr is None:
            continue
        correlations.append((key, corr))
    if correlations:
        correlations.sort(key=lambda item: abs(item[1]), reverse=True)

    top_offcpu = sorted(
        (
            (pid, total_ns / 1_000_000.0)
            for pid, total_ns in system_data.total_offcpu_ns.items()
            if total_ns > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    top_pagefaults = sorted(
        (
            (pid, count)
            for pid, count in system_data.total_page_faults.items()
            if count > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    return {
        "correlations": correlations,
        "time_series": time_series,
        "top_offcpu": top_offcpu,
        "top_pagefaults": top_pagefaults,
        "aligned_pairs": len(time_series),
    }


def _prepare_stats() -> Dict[int, DeviceStats]:
    return {}


def _collect_stats(
    snapshot: Dict[str, object],
    devices: Dict[int, DeviceStats],
    global_latencies: List[LatencySnapshot],
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

    combined_buckets: Dict[int, int] = {}
    combined_total = 0

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
        combined_total += total_requests
        for slot, value in snapshot_buckets.items():
            combined_buckets[slot] = combined_buckets.get(slot, 0) + value

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
    if not math.isnan(timestamp) and combined_buckets:
        if combined_total <= 0:
            combined_total = sum(combined_buckets.values())
        global_latencies.append(
            LatencySnapshot(
                timestamp=timestamp,
                interval_s=interval_s,
                total_requests=combined_total,
                bucket_counts=combined_buckets,
            )
        )


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
    importance_payload: Optional[Dict[str, object]] = None,
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

    if "importance" in enabled_charts and importance_payload:
        LABELS = {
            "off_cpu_ms": "Tiempo off-CPU (ms)",
            "page_faults_per_s": "Fallos de página /s",
            "cpu_pressure": "PSI CPU (avg)",
            "io_pressure": "PSI I/O (avg)",
            "memory_pressure": "PSI Memoria (avg)",
        }
        correlations = importance_payload.get("correlations", [])
        if correlations:
            names = [LABELS.get(key, key) for key, _ in correlations]
            values = [corr for _, corr in correlations]
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#1f77b4" if val >= 0 else "#d62728" for val in values]
            ax.bar(range(len(values)), values, color=colors, alpha=0.8)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Correlación (Pearson)")
            aligned_pairs = importance_payload.get("aligned_pairs", 0)
            ax.set_title(
                f"Importancia relativa vs latencia (pares alineados: {aligned_pairs})"
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(
                charts_dir / "importance_correlations.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        top_offcpu = importance_payload.get("top_offcpu", [])
        if top_offcpu:
            pids = [pid for pid, _ in top_offcpu]
            values_ms = [value for _, value in top_offcpu]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(range(len(values_ms)), values_ms, color="#9467bd")
            ax.set_yticks(range(len(values_ms)))
            ax.set_yticklabels(pids, fontsize=9)
            ax.set_xlabel("Tiempo acumulado fuera de CPU (ms)")
            ax.set_title("PIDs con mayor tiempo off-CPU")
            ax.grid(True, axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(
                charts_dir / "importance_offcpu_top_pids.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        top_pagefaults = importance_payload.get("top_pagefaults", [])
        if top_pagefaults:
            pids = [pid for pid, _ in top_pagefaults]
            values_pf = [value for _, value in top_pagefaults]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(range(len(values_pf)), values_pf, color="#ff7f0e")
            ax.set_yticks(range(len(values_pf)))
            ax.set_yticklabels(pids, fontsize=9)
            ax.set_xlabel("Fallos de página acumulados")
            ax.set_title("PIDs con más fallos de página")
            ax.grid(True, axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(
                charts_dir / "importance_page_faults_top_pids.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        time_series = importance_payload.get("time_series", [])
        if time_series:
            times = [entry.get("time", 0.0) for entry in time_series]
            latency_values = [entry.get("latency_us", math.nan) for entry in time_series]
            offcpu_values = [entry.get("off_cpu_ms", math.nan) for entry in time_series]
            pf_values = [
                entry.get("page_faults_per_s", math.nan) for entry in time_series
            ]
            cpu_pressure_values = [
                entry.get("cpu_pressure", math.nan) for entry in time_series
            ]
            io_pressure_values = [
                entry.get("io_pressure", math.nan) for entry in time_series
            ]
            mem_pressure_values = [
                entry.get("memory_pressure", math.nan) for entry in time_series
            ]

            def _normalize(series: List[float]) -> List[float]:
                finite = [value for value in series if not math.isnan(value)]
                if not finite:
                    return [math.nan for _ in series]
                min_v = min(finite)
                max_v = max(finite)
                if math.isclose(max_v, min_v):
                    return [0.5 for _ in series]
                return [
                    (value - min_v) / (max_v - min_v) if not math.isnan(value) else math.nan
                    for value in series
                ]

            latency_norm = _normalize(latency_values)
            offcpu_norm = _normalize(offcpu_values)
            pf_norm = _normalize(pf_values)
            cpu_pressure_norm = _normalize(cpu_pressure_values)
            io_pressure_norm = _normalize(io_pressure_values)
            mem_pressure_norm = _normalize(mem_pressure_values)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(times, latency_norm, label="Latencia (normalizada)", linewidth=1.8)
            ax.plot(times, offcpu_norm, label="Off-CPU ms (norm)", linewidth=1.2)
            ax.plot(
                times,
                pf_norm,
                label="Fallos de página/s (norm)",
                linewidth=1.2,
                linestyle="--",
            )
            ax.plot(
                times,
                cpu_pressure_norm,
                label="PSI CPU (norm)",
                linewidth=1.0,
                linestyle=":",
            )
            ax.plot(
                times,
                io_pressure_norm,
                label="PSI I/O (norm)",
                linewidth=1.0,
                linestyle="-.",
            )
            ax.plot(
                times,
                mem_pressure_norm,
                label="PSI Memoria (norm)",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
            ax.set_xlabel("Tiempo desde el inicio (s)")
            ax.set_ylabel("Escala normalizada (0-1)")
            ax.set_title("Tendencias normalizadas de recursos vs latencia")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper right", ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(
                charts_dir / "importance_trends.png",
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
    global_snapshots: List[LatencySnapshot] = []
    snapshots = 0

    for snapshot in _read_snapshots(args.input):
        if not isinstance(snapshot, dict):
            continue
        _collect_stats(snapshot, devices, global_snapshots)
        snapshots += 1

    if snapshots == 0:
        print("No se encontraron snapshots en el archivo.")
        return

    system_metrics_data = None
    importance_payload = None
    if args.system_metrics:
        system_metrics_data = _read_system_metrics(args.system_metrics)
        importance_payload = _compute_importance_payload(
            global_snapshots, system_metrics_data, percentiles
        )
        if args.system_metrics.exists() and system_metrics_data is None:
            print(
                f"Advertencia: no se pudieron leer snapshots de {args.system_metrics}.",
                file=sys.stderr,
            )
        if (
            not importance_payload
            and args.system_metrics.exists()
            and "importance" in chart_types
        ):
            print(
                "Advertencia: no se generaron gráficas de importancia (no hay datos alineados suficientes).",
                file=sys.stderr,
            )

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
        _generate_charts(
            devices,
            args.charts_dir,
            percentiles,
            chart_types,
            importance_payload,
        )
        print(f"Gráficas exportadas a: {args.charts_dir}")


if __name__ == "__main__":
    main()
