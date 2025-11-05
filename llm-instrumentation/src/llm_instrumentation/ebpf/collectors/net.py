from __future__ import annotations

import time
from typing import Any, Dict, Iterator

from .base import BaseCollector, CollectorConfig


class NetworkCollector(BaseCollector):
    """Collect TCP send/receive throughput and retransmissions."""

    NAME = "net"

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__(config)
        self._last_emit_ns = time.time_ns()

    def build_bpf(self) -> str:
        return r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct net_stat_t {
    u64 send_bytes;
    u64 recv_bytes;
    u64 send_calls;
    u64 recv_calls;
    u64 retransmits;
    char comm[TASK_COMM_LEN];
};

BPF_HASH(net_stats, u32, struct net_stat_t);

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

static inline struct net_stat_t *get_stat(void) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    struct net_stat_t *stat = net_stats.lookup(&pid);
    if (!stat) {
        struct net_stat_t zero = {};
        net_stats.update(&pid, &zero);
        stat = net_stats.lookup(&pid);
    }
    if (stat) {
        bpf_get_current_comm(stat->comm, sizeof(stat->comm));
    }
    return stat;
}

TRACEPOINT_PROBE(tcp, tcp_sendmsg)
{
    if (!pass_filters()) {
        return 0;
    }
    struct net_stat_t *stat = get_stat();
    if (!stat) {
        return 0;
    }
    stat->send_bytes += args->size;
    stat->send_calls += 1;
    return 0;
}

TRACEPOINT_PROBE(tcp, tcp_cleanup_rbuf)
{
    if (!pass_filters()) {
        return 0;
    }
    struct net_stat_t *stat = get_stat();
    if (!stat) {
        return 0;
    }
    stat->recv_bytes += args->copied;
    stat->recv_calls += 1;
    return 0;
}

TRACEPOINT_PROBE(tcp, tcp_retransmit_skb)
{
    if (!pass_filters()) {
        return 0;
    }
    struct net_stat_t *stat = get_stat();
    if (!stat) {
        return 0;
    }
    stat->retransmits += 1;
    return 0;
}
"""

    def do_attach(self, bpf: Any) -> None:
        del bpf

    def consume(self) -> Iterator[Dict[str, Any]]:
        now_ns = time.time_ns()
        elapsed = max(1e-9, (now_ns - self._last_emit_ns) / 1e9)
        self._last_emit_ns = now_ns

        stats_map = self.bpf["net_stats"]

        for key, value in stats_map.items():
            pid = key.value
            comm = bytes(value.comm).split(b"\x00", 1)[0].decode("utf-8", "replace")
            send_bytes = int(value.send_bytes)
            recv_bytes = int(value.recv_bytes)
            yield {
                "type": "net",
                "pid": pid,
                "comm": comm,
                "tx_bytes_s": send_bytes / elapsed,
                "rx_bytes_s": recv_bytes / elapsed,
                "tx_calls": int(value.send_calls),
                "rx_calls": int(value.recv_calls),
                "retransmits": int(value.retransmits),
            }
        stats_map.clear()
