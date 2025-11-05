from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterator, List

from .base import BaseCollector, CollectorConfig


class SchedulerCollector(BaseCollector):
    """Measure off-CPU time and runqueue latency using sched tracepoints."""

    NAME = "sched"

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__(config)
        self._last_emit_ns = time.time_ns()

    def build_bpf(self) -> str:
        return r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct start_t {
    u64 ts;
    u32 tgid;
    char comm[TASK_COMM_LEN];
};

struct stat_t {
    u64 offcpu_ns;
    u64 switches;
    u32 tgid;
    char comm[TASK_COMM_LEN];
    u64 wait_ns_total;
    u64 wait_ns_max;
};

struct hist_key_t {
    u32 tid;
    u64 slot;
};

BPF_HASH(offcpu_start, u32, struct start_t);
BPF_HASH(offcpu_stats, u32, struct stat_t);
BPF_HASH(wakeup_ts, u32, u64);
BPF_HISTOGRAM(runq_hist, struct hist_key_t);

static inline bool pass_filters(u32 tgid, char comm[TASK_COMM_LEN]) {
#ifdef TARGET_PID
    if (tgid != TARGET_PID) {
        return false;
    }
#endif
#ifdef TARGET_COMM
    const char target[] = TARGET_COMM;
    if (__builtin_memcmp(comm, target, sizeof(target) - 1) != 0) {
        return false;
    }
#endif
#ifdef TARGET_CGROUP_ID
    u64 cgid = bpf_get_current_cgroup_id();
    if (cgid != TARGET_CGROUP_ID) {
        return false;
    }
#endif
    return true;
}

TRACEPOINT_PROBE(sched, sched_switch)
{
    u64 ts = bpf_ktime_get_ns();
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 current_tid = pid_tgid;
    u32 current_tgid = pid_tgid >> 32;

    char current_comm[TASK_COMM_LEN];
    bpf_get_current_comm(current_comm, sizeof(current_comm));

    if (current_tid && pass_filters(current_tgid, current_comm)) {
        struct start_t start = {
            .ts = ts,
            .tgid = current_tgid,
        };
        __builtin_memcpy(start.comm, current_comm, sizeof(current_comm));
        offcpu_start.update(&current_tid, &start);
    }

    u32 next_tid = args->next_pid;
    if (next_tid == 0) {
        return 0;
    }

    struct start_t *start = offcpu_start.lookup(&next_tid);
    if (!start) {
        wakeup_ts.delete(&next_tid);
        return 0;
    }

    u64 delta = ts - start->ts;
    struct stat_t *stat = offcpu_stats.lookup(&next_tid);
    if (!stat) {
        struct stat_t zero = {};
        offcpu_stats.update(&next_tid, &zero);
        stat = offcpu_stats.lookup(&next_tid);
    }
    if (stat) {
        stat->offcpu_ns += delta;
        stat->switches += 1;
        stat->tgid = start->tgid;
        __builtin_memcpy(stat->comm, start->comm, TASK_COMM_LEN);
    }
    offcpu_start.delete(&next_tid);

    u64 *wake = wakeup_ts.lookup(&next_tid);
    if (wake) {
        u64 wait = ts - *wake;
        if (stat) {
            stat->wait_ns_total += wait;
            if (wait > stat->wait_ns_max) {
                stat->wait_ns_max = wait;
            }
        }
        u64 usec = wait / 1000;
        if (usec == 0) {
            usec = 1;
        }
        struct hist_key_t hkey = {
            .tid = next_tid,
            .slot = bpf_log2l(usec),
        };
        runq_hist.atomic_increment(hkey);
        wakeup_ts.delete(&next_tid);
    }

    return 0;
}

TRACEPOINT_PROBE(sched, sched_wakeup)
{
    u32 tid = args->pid;
    u64 ts = bpf_ktime_get_ns();
    wakeup_ts.update(&tid, &ts);
    return 0;
}
"""

    def do_attach(self, bpf: Any) -> None:
        del bpf

    def consume(self) -> Iterator[Dict[str, Any]]:
        now_ns = time.time_ns()
        elapsed_s = max(1e-9, (now_ns - self._last_emit_ns) / 1e9)
        self._last_emit_ns = now_ns

        stats_map = self.bpf["offcpu_stats"]
        hist_map = self.bpf["runq_hist"]

        histograms: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
        for key, value in hist_map.items():
            tid = key.tid
            histograms[tid][key.slot] = histograms[tid].get(key.slot, 0) + value.value
        hist_map.clear()

        threads: List[Dict[str, Any]] = []
        total_offcpu_pct = 0.0

        for key, value in stats_map.items():
            tid = key.value
            tgid = int(value.tgid)
            if self.config.pid and tgid != self.config.pid:
                continue
            count = int(value.switches) if value.switches else 1
            offcpu_ns = int(value.offcpu_ns)
            wait_total = int(value.wait_ns_total)
            wait_max = int(value.wait_ns_max)
            comm = bytes(value.comm).split(b"\x00", 1)[0].decode("utf-8", "replace")

            offcpu_pct = (offcpu_ns / (elapsed_s * 1e9)) * 100.0
            total_offcpu_pct += offcpu_pct
            wait_hist = histograms.get(tid, {})
            rq_percentiles = _percentiles_from_hist(wait_hist)

            threads.append(
                {
                    "tid": tid,
                    "tgid": tgid,
                    "comm": comm,
                    "off_cpu_pct": offcpu_pct,
                    "off_cpu_ns": offcpu_ns,
                    "switches": count,
                    "rq_latency_p95_us": rq_percentiles.get("p95"),
                    "rq_latency_p99_us": rq_percentiles.get("p99"),
                    "rq_latency_max_us": wait_max / 1000.0,
                }
            )

        stats_map.clear()

        threads.sort(key=lambda item: item["off_cpu_pct"], reverse=True)

        yield {
            "type": "sched",
            "total_off_cpu_pct": total_offcpu_pct,
            "threads": threads[:10],
        }


def _percentiles_from_hist(buckets: Dict[int, int]) -> Dict[str, float]:
    total = sum(buckets.values())
    if not buckets or total <= 0:
        return {"p95": 0.0, "p99": 0.0}
    thresholds = {"p95": total * 0.95, "p99": total * 0.99}
    results = {key: 0.0 for key in thresholds}
    cumulative = 0.0
    for slot in sorted(buckets):
        cumulative += buckets[slot]
        value_us = float(1 << slot)
        for name, limit in thresholds.items():
            if results[name] == 0.0 and cumulative >= limit:
                results[name] = value_us
    return results
