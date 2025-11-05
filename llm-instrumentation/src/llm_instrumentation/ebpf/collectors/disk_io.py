from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterator, List, Tuple

from .base import BaseCollector, CollectorConfig


class DiskIOCollector(BaseCollector):
    """
    Collects disk latency histograms, bytes, and queue depth using block tracepoints.
    """

    NAME = "disk"

    REQUIRED_TRACEPOINTS = (
        ("block", "block_rq_issue"),
        ("block", "block_rq_complete"),
    )

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__(config)
        self._last_emit_ns = time.time_ns()

    def build_bpf(self) -> str:
        return r"""
// disk_io.py generated program
#include <uapi/linux/ptrace.h>
#include <linux/blk-mq.h>
#include <linux/blkdev.h>
#include <linux/sched.h>

#ifndef SAMPLE_EVERY
#define SAMPLE_EVERY 1
#endif

struct rq_key_t {
    u64 req;
};

struct rq_start_t {
    u64 ts;
    dev_t dev;
    u32 bytes;
};

struct queue_stat_t {
    u64 depth;
    u64 max_depth;
    u64 sum_depth;
    u64 samples;
};

struct io_stat_t {
    u64 bytes;
    u64 requests;
};

struct hist_key_t {
    dev_t dev;
    u64 slot;
};

BPF_HASH(start, struct rq_key_t, struct rq_start_t);
BPF_HASH(queue_stats, dev_t, struct queue_stat_t);
BPF_HASH(io_stats, dev_t, struct io_stat_t);
BPF_HISTOGRAM(latency_hist, struct hist_key_t);

static inline bool pass_filters(void) {
#ifdef TARGET_PID
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != TARGET_PID) {
        return false;
    }
#endif
#ifdef TARGET_CGROUP_ID
    u64 cgid = bpf_get_current_cgroup_id();
    if (cgid != TARGET_CGROUP_ID) {
        return false;
    }
#endif
#ifdef TARGET_COMM
    char comm[TASK_COMM_LEN];
    bpf_get_current_comm(comm, sizeof(comm));
    const char target[] = TARGET_COMM;
    if (__builtin_memcmp(comm, target, sizeof(target) - 1) != 0) {
        return false;
    }
#endif
    return true;
}

TRACEPOINT_PROBE(block, block_rq_issue)
{
    if (!pass_filters()) {
        return 0;
    }

    struct rq_key_t key = {
        .req = (u64)args->rq,
    };

    struct rq_start_t val = {
        .ts = bpf_ktime_get_ns(),
        .dev = args->dev,
        .bytes = args->nr_bytes,
    };
    start.update(&key, &val);

    dev_t dev = args->dev;
    struct queue_stat_t *q = queue_stats.lookup(&dev);
    if (!q) {
        struct queue_stat_t zero = {};
        queue_stats.update(&dev, &zero);
        q = queue_stats.lookup(&dev);
        if (!q) {
            return 0;
        }
    }
    q->depth += 1;
    q->sum_depth += q->depth;
    q->samples += 1;
    if (q->depth > q->max_depth) {
        q->max_depth = q->depth;
    }
    return 0;
}

TRACEPOINT_PROBE(block, block_rq_complete)
{
    struct rq_key_t key = {
        .req = (u64)args->rq,
    };

    struct rq_start_t *start_ts = start.lookup(&key);
    if (!start_ts) {
        return 0;
    }

    u64 delta = bpf_ktime_get_ns() - start_ts->ts;
    dev_t dev = start_ts->dev;
    u32 bytes = start_ts->bytes;
    start.delete(&key);

    if (delta == 0) {
        delta = 1;
    }

    struct hist_key_t hkey = {
        .dev = dev,
        .slot = bpf_log2l(delta / 1000),
    };
    latency_hist.atomic_increment(hkey);

    struct io_stat_t *io = io_stats.lookup(&dev);
    if (!io) {
        struct io_stat_t zero = {};
        io_stats.update(&dev, &zero);
        io = io_stats.lookup(&dev);
        if (!io) {
            return 0;
        }
    }
    io->bytes += bytes;
    io->requests += 1;

    struct queue_stat_t *q = queue_stats.lookup(&dev);
    if (q && q->depth > 0) {
        q->depth -= 1;
    }
    return 0;
}
"""

    def do_attach(self, bpf: Any) -> None:
        # Tracepoint probes auto-attached by BCC when TRACEPOINT_PROBE macro is used.
        del bpf  # nothing extra to do

    def consume(self) -> Iterator[Dict[str, Any]]:
        now_ns = time.time_ns()
        elapsed = max(1e-9, (now_ns - self._last_emit_ns) / 1e9)
        self._last_emit_ns = now_ns

        hist = self.bpf["latency_hist"]
        io_stats = self.bpf["io_stats"]
        queue_stats = self.bpf["queue_stats"]

        histogram: DefaultDict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for key, value in hist.items():
            dev = key.dev
            bucket = key.slot
            histogram[dev][bucket] += value.value
        hist.clear()

        io_totals: Dict[int, Tuple[int, int]] = {}
        for key, value in io_stats.items():
            io_totals[key.value] = (value.bytes, value.requests)
        io_stats.clear()

        queue_totals: Dict[int, Dict[str, float]] = {}
        Leaf = queue_stats.Leaf  # type: ignore[attr-defined]
        for key, value in queue_stats.items():
            depth = float(value.depth)
            avg_depth = 0.0
            if value.samples:
                avg_depth = value.sum_depth / float(value.samples)
            queue_totals[key.value] = {
                "avg": avg_depth,
                "max": float(value.max_depth),
            }
            new_leaf = Leaf()
            new_leaf.depth = int(depth)
            new_leaf.max_depth = 0
            new_leaf.sum_depth = 0
            new_leaf.samples = 0
            queue_stats[key] = new_leaf

        for dev, buckets in histogram.items():
            bytes_total, request_total = io_totals.get(dev, (0, 0))
            device_name = _resolve_device_name(dev)
            percentiles = _percentiles_from_hist(buckets, request_total)
            qdepth = queue_totals.get(dev, {"avg": 0.0, "max": 0})

            iops = request_total / elapsed
            throughput = bytes_total / elapsed

            yield {
                "type": "disk",
                "dev": device_name,
                "iops": iops,
                "throughput_bytes_s": throughput,
                "lat_p50_us": percentiles.get("p50"),
                "lat_p95_us": percentiles.get("p95"),
                "lat_p99_us": percentiles.get("p99"),
                "queue_depth_avg": qdepth["avg"],
                "queue_depth_max": qdepth["max"],
                "requests": request_total,
                "bytes": bytes_total,
            }


def _resolve_device_name(dev: int) -> str:
    major = os.major(dev)
    minor = os.minor(dev)
    path = f"/sys/dev/block/{major}:{minor}"
    try:
        target = os.readlink(path)
        return target.split("/")[-1]
    except OSError:
        return f"{major}:{minor}"


def _percentiles_from_hist(
    buckets: Dict[int, int], total: int
) -> Dict[str, float]:
    if total == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    thresholds = {
        "p50": total * 0.5,
        "p95": total * 0.95,
        "p99": total * 0.99,
    }
    results = {key: 0.0 for key in thresholds}
    cumulative = 0.0

    for slot in sorted(buckets):
        cumulative += buckets[slot]
        value_us = float(1 << slot)
        for name, limit in thresholds.items():
            if results[name] == 0.0 and cumulative >= limit:
                results[name] = value_us

    return results
