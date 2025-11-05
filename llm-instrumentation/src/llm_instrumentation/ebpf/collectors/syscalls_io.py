from __future__ import annotations

import platform
import time
import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterator, List

try:
    from bcc import syscall as bcc_syscall  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on user environment
    bcc_syscall = None

from .base import BaseCollector, CollectorConfig


_FALLBACK_SYSCALLS_X86_64: Dict[int, str] = {
    0: "read",
    1: "write",
    3: "close",
    7: "poll",
    9: "mmap",
    11: "munmap",
    17: "pread64",
    18: "pwrite64",
    42: "connect",
    43: "accept",
    44: "sendto",
    45: "recvfrom",
    46: "sendmsg",
    47: "recvmsg",
    74: "fsync",
    202: "futex",
    230: "clock_nanosleep",
    232: "epoll_wait",
    257: "openat",
    271: "ppoll",
    285: "fallocate",
}


def _load_syscall_table() -> Dict[int, str]:
    if bcc_syscall and hasattr(bcc_syscall, "syscalls"):
        result: Dict[int, str] = {}
        for nr, name in bcc_syscall.syscalls.items():
            decoded = name.decode("utf-8")
            result[nr] = decoded
        return result

    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64", "x64"):
        warnings.warn(
            "bcc.syscall is unavailable; falling back to a static syscall table for x86_64. "
            "Install the full BCC python bindings to use dynamic tables.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(_FALLBACK_SYSCALLS_X86_64)

    raise RuntimeError(
        "Unable to load syscall table: bcc.syscall module missing and no fallback available "
        f"for architecture '{arch}'. Install the BCC python bindings (e.g. `python-bcc`) to "
        "enable syscall collectors."
    )


ALL_SYSCALLS: Dict[int, str] = _load_syscall_table()
DEFAULT_TRACKED_NAMES = [
    "read",
    "write",
    "pread64",
    "pwrite64",
    "openat",
    "close",
    "fsync",
    "fallocate",
    "mmap",
    "munmap",
    "sendto",
    "sendmsg",
    "recvfrom",
    "recvmsg",
    "connect",
    "accept",
]


class SyscallLatencyCollector(BaseCollector):
    """Collects syscall latency and volume information using raw_syscalls tracepoints."""

    NAME = "syscall"

    def __init__(self, config: CollectorConfig, tracked_names: List[str] | None = None) -> None:
        super().__init__(config)
        self._last_emit_ns = time.time_ns()
        names = tracked_names or DEFAULT_TRACKED_NAMES
        tracked: Dict[int, str] = {}
        for nr, name in ALL_SYSCALLS.items():
            if name in names:
                tracked[nr] = name
        if not tracked:
            raise ValueError("No tracked syscalls available on this architecture.")
        self._tracked_syscalls = tracked

    def build_bpf(self) -> str:
        tracked_cases = "\n".join(
            f"    case {nr}: return true;" for nr in self._tracked_syscalls.keys()
        )
        bytes_switch = "\n".join(
            _byte_case(nr, name) for nr, name in self._tracked_syscalls.items()
        )
        return f"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/unistd.h>

#ifndef SAMPLE_EVERY
#define SAMPLE_EVERY 1
#endif

struct active_t {{
    u64 ts;
    u32 id;
    u64 req_bytes;
    u64 pid;
}};

struct stat_t {{
    u64 count;
    u64 total_ns;
    u64 max_ns;
    u64 bytes;
}};

struct hist_key_t {{
    u32 id;
    u64 slot;
}};

struct caller_key_t {{
    u32 id;
    u32 pid;
    char comm[TASK_COMM_LEN];
}};

struct caller_val_t {{
    u64 count;
    u64 total_ns;
    u64 bytes;
}};

BPF_HASH(active, u32, struct active_t);
BPF_HASH(stats, u32, struct stat_t);
BPF_HISTOGRAM(latency_hist, struct hist_key_t);
BPF_HASH(callers, struct caller_key_t, struct caller_val_t);

static inline bool is_tracked(u32 id) {{
    switch (id) {{
{tracked_cases}
    default:
        return false;
    }}
}}

static inline bool should_sample(u32 id) {{
#ifdef SAMPLE_EVERY
    if (SAMPLE_EVERY <= 1) {{
        return true;
    }}
#endif
    switch (id) {{
        case __NR_read:
        case __NR_write:
        case __NR_pread64:
        case __NR_pwrite64:
        case __NR_recvfrom:
        case __NR_sendto:
        case __NR_recvmsg:
        case __NR_sendmsg:
            return bpf_get_prandom_u32() % SAMPLE_EVERY == 0;
    }}
    return true;
}}

static inline bool pass_filters(void) {{
#ifdef TARGET_PID
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != TARGET_PID) {{
        return false;
    }}
#endif
#ifdef TARGET_CGROUP_ID
    u64 cgid = bpf_get_current_cgroup_id();
    if (cgid != TARGET_CGROUP_ID) {{
        return false;
    }}
#endif
#ifdef TARGET_COMM
    char comm[TASK_COMM_LEN];
    bpf_get_current_comm(comm, sizeof(comm));
    const char target[] = TARGET_COMM;
    if (__builtin_memcmp(comm, target, sizeof(target) - 1) != 0) {{
        return false;
    }}
#endif
    return true;
}}

static inline u64 requested_bytes(u32 id, struct trace_event_raw_sys_enter *ctx) {{
    switch (id) {{
{bytes_switch}
    default:
        return 0;
    }}
}}

TRACEPOINT_PROBE(raw_syscalls, sys_enter)
{{
    u32 id = ctx->id;
    if (!is_tracked(id)) {{
        return 0;
    }}
    if (!pass_filters()) {{
        return 0;
    }}
    if (!should_sample(id)) {{
        return 0;
    }}
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct active_t val = {{
        .ts = bpf_ktime_get_ns(),
        .id = id,
        .req_bytes = requested_bytes(id, ctx),
        .pid = pid_tgid >> 32,
    }};
    active.update(&tid, &val);
    return 0;
}}

TRACEPOINT_PROBE(raw_syscalls, sys_exit)
{{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tid = pid_tgid;
    struct active_t *entry = active.lookup(&tid);
    if (!entry) {{
        return 0;
    }}
    u64 delta = bpf_ktime_get_ns() - entry->ts;
    u32 id = entry->id;
    active.delete(&tid);

    struct stat_t *stat = stats.lookup(&id);
    if (!stat) {{
        struct stat_t zero = {{}};
        stats.update(&id, &zero);
        stat = stats.lookup(&id);
    }}
    if (stat) {{
        stat->count += 1;
        stat->total_ns += delta;
        if (delta > stat->max_ns) {{
            stat->max_ns = delta;
        }}
        long ret = ctx->ret;
        if (ret > 0) {{
            stat->bytes += ret;
        }} else {{
            stat->bytes += entry->req_bytes;
        }}
    }}

    u64 usec = delta / 1000;
    if (usec == 0) {{
        usec = 1;
    }}
    struct hist_key_t hkey = {{
        .id = id,
        .slot = bpf_log2l(usec),
    }};
    latency_hist.atomic_increment(hkey);

    struct caller_key_t ckey = {{
        .id = id,
        .pid = entry->pid,
    }};
    bpf_get_current_comm(ckey.comm, sizeof(ckey.comm));
    struct caller_val_t *cval = callers.lookup(&ckey);
    if (!cval) {{
        struct caller_val_t zero = {{}};
        callers.update(&ckey, &zero);
        cval = callers.lookup(&ckey);
    }}
    if (cval) {{
        cval->count += 1;
        cval->total_ns += delta;
        long ret = ctx->ret;
        if (ret > 0) {{
            cval->bytes += ret;
        }} else {{
            cval->bytes += entry->req_bytes;
        }}
    }}
    return 0;
}}
"""

    def do_attach(self, bpf: Any) -> None:
        # TRACEPOINT_PROBE auto-attached; nothing else to do.
        del bpf

    def consume(self) -> Iterator[Dict[str, Any]]:
        now_ns = time.time_ns()
        elapsed = max(1e-9, (now_ns - self._last_emit_ns) / 1e9)
        self._last_emit_ns = now_ns

        stats_map = self.bpf["stats"]
        hist_map = self.bpf["latency_hist"]
        callers_map = self.bpf["callers"]

        histograms: DefaultDict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for key, value in hist_map.items():
            histograms[key.id][key.slot] += value.value
        hist_map.clear()

        callers: DefaultDict[int, List[Dict[str, object]]] = defaultdict(list)
        for key, value in callers_map.items():
            callers[key.id].append(
                {
                    "pid": key.pid,
                    "comm": bytes(key.comm)
                    .split(b"\x00", 1)[0]
                    .decode("utf-8", "replace"),
                    "count": int(value.count),
                    "lat_total_ns": int(value.total_ns),
                    "bytes": int(value.bytes),
                }
            )
        callers_map.clear()

        for key, value in stats_map.items():
            syscall_id = key.value
            count = int(value.count)
            total_ns = int(value.total_ns)
            max_ns = int(value.max_ns)
            bytes_total = int(value.bytes)
            name = self._tracked_syscalls.get(syscall_id, f"sys_{syscall_id}")
            hist = histograms.get(syscall_id, {})
            percentiles = _percentiles_from_hist(hist, count)
            avg_ns = total_ns / count if count else 0.0
            bytes_per_call = bytes_total / count if count else 0.0
            iops = count / elapsed
            event = {
                "type": "syscall",
                "syscall": name,
                "count": count,
                "lat_avg_us": avg_ns / 1000.0,
                "lat_p95_us": percentiles.get("p95"),
                "lat_p99_us": percentiles.get("p99"),
                "lat_max_us": max_ns / 1000.0,
                "bytes": bytes_total,
                "bytes_per_call": bytes_per_call,
                "throughput_bytes_s": bytes_total / elapsed,
                "top_callers": sorted(
                    callers.get(syscall_id, []),
                    key=lambda item: item["lat_total_ns"],
                    reverse=True,
                )[:5],
            }
            yield event
        stats_map.clear()


def _byte_case(nr: int, name: str) -> str:
    arg_index = {
        "read": 2,
        "write": 2,
        "pread64": 2,
        "pwrite64": 2,
        "fallocate": 3,
        "mmap": 1,
        "munmap": 1,
        "sendto": 2,
        "recvfrom": 2,
        "sendmsg": -1,
        "recvmsg": -1,
    }.get(name, -1)
    if arg_index < 0:
        return f"    case {nr}: return 0;"
    return f"    case {nr}: return ctx->args[{arg_index}];"


def _percentiles_from_hist(buckets: Dict[int, int], total: int) -> Dict[str, float]:
    if total == 0:
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
