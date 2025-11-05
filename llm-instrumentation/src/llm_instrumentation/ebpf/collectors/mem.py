from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from .base import BaseCollector, CollectorConfig


class MemoryPressureCollector(BaseCollector):
    """Collect user page faults per PID/TID and augment with PSI and RSS."""

    NAME = "mem"

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__(config)
        self._last_counts: Dict[Tuple[int, int], int] = {}

    def build_bpf(self) -> str:
        return r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct fault_key_t {
    u32 pid;
    u32 tid;
};

BPF_HASH(page_faults, struct fault_key_t, u64);

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

TRACEPOINT_PROBE(exceptions, page_fault_user)
{
    if (!pass_filters()) {
        return 0;
    }

    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct fault_key_t key = {
        .pid = pid_tgid >> 32,
        .tid = pid_tgid,
    };

    u64 *count = page_faults.lookup(&key);
    if (!count) {
        u64 one = 1;
        page_faults.update(&key, &one);
    } else {
        __sync_fetch_and_add(count, 1);
    }
    return 0;
}
"""

    def do_attach(self, bpf: Any) -> None:
        del bpf

    def consume(self) -> Iterator[Dict[str, Any]]:
        faults_map = self.bpf["page_faults"]
        totals: Dict[int, int] = {}
        top_threads: List[Dict[str, int]] = []
        new_counts: Dict[Tuple[int, int], int] = {}
        delta_total = 0

        for key, value in faults_map.items():
            pid = key.pid
            tid = key.tid
            count = int(value.value)
            new_counts[(pid, tid)] = count
            prev = self._last_counts.get((pid, tid), 0)
            delta = max(0, count - prev)
            delta_total += delta
            totals[pid] = totals.get(pid, 0) + delta
            if delta > 0:
                top_threads.append({"pid": pid, "tid": tid, "faults": delta})

        self._last_counts = new_counts
        top_threads.sort(key=lambda item: item["faults"], reverse=True)

        psi = {
            domain: _read_psi(domain) for domain in ("cpu", "io", "memory")
        }

        rss_kb = None
        major_faults = None
        if self.config.pid:
            rss_kb = _read_rss_kb(self.config.pid)
            major_faults = _read_major_faults(self.config.pid)

        yield {
            "type": "mem",
            "page_faults_total": delta_total,
            "page_faults_by_pid": totals,
            "top_faulting_threads": top_threads[:5],
            "psi": psi,
            "rss_kb": rss_kb,
            "major_faults": major_faults,
        }


def _read_psi(domain: str) -> Dict[str, float]:
    path = Path("/proc/pressure") / domain
    data = {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return data
    for line in text.splitlines():
        parts = line.split()
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if key in data:
                try:
                    data[key] = float(value)
                except ValueError:
                    pass
    return data


def _read_rss_kb(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except FileNotFoundError:
        return None
    return None


def _read_major_faults(pid: int) -> int | None:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        content = stat_path.read_text(encoding="utf-8")
        fields = content.split()
        if len(fields) > 11:
            return int(fields[11])
    except FileNotFoundError:
        return None
    except ValueError:
        return None
    return None
