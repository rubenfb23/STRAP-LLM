"""
`llm-prof` command line interface that orchestrates ebpf collectors.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

from ..ebpf import (
    AggregationConfig,
    DiskIOCollector,
    IntervalAggregator,
    MemoryPressureCollector,
    NetworkCollector,
    SchedulerCollector,
    SyscallLatencyCollector,
    SyncWaitCollector,
)
from ..ebpf.collectors import BaseCollector, CollectorConfig
from ..ebpf.correlation import CorrelationContext


LOG = logging.getLogger("llm_prof")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llm-prof",
        description="Low-overhead runtime profiler for LLM inference pipelines using eBPF collectors.",
    )
    parser.add_argument("--pid", type=int, help="PID to filter events by.")
    parser.add_argument("--cgroup", type=str, help="cgroup v2 path relative to /sys/fs/cgroup.")
    parser.add_argument("--comm", type=str, help="Restrict events by command name.")
    parser.add_argument("--interval", type=float, default=1.0, help="Aggregation interval seconds.")
    parser.add_argument("--output", type=Path, help="Write JSONL to this path.")
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable file output and only stream aggregates to stdout.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5,
        help="Flush after this many events when writing to disk.",
    )
    parser.add_argument(
        "--fsync",
        action="store_true",
        help="Force fsync after each flush (higher durability with added overhead).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="disk,syscalls,mem,sched",
        help="Comma separated collectors to enable: disk,syscalls,sync,mem,sched,net.",
    )
    parser.add_argument(
        "--sample-read-write",
        type=int,
        default=1,
        help="Sample ratio for heavy syscalls (1 = capture all, 10 = 1/10).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Comma separated key=value pairs added to every event for filtering.",
    )
    parser.add_argument(
        "--disable-correlation",
        action="store_true",
        help="Do not propagate correlation IDs from the application context.",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Optional subcommand. Currently only `top` is supported for in-CLI summaries.",
    )
    return parser.parse_args(argv)


def build_collectors(args: argparse.Namespace) -> List[BaseCollector]:
    config = CollectorConfig(
        pid=args.pid,
        cgroup_path=args.cgroup,
        comm_filter=args.comm,
        sample_every=max(1, args.sample_read_write),
    )
    enabled = {item.strip() for item in args.only.split(",") if item.strip()}
    collectors: List[BaseCollector] = []

    if "disk" in enabled:
        collectors.append(DiskIOCollector(config))
    if "syscalls" in enabled:
        collectors.append(SyscallLatencyCollector(config))
    if "sync" in enabled:
        collectors.append(SyncWaitCollector(config))
    if "mem" in enabled:
        collectors.append(MemoryPressureCollector(config))
    if "sched" in enabled:
        collectors.append(SchedulerCollector(config))
    if "net" in enabled:
        collectors.append(NetworkCollector(config))

    return collectors


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    out: dict[str, str] = {}
    for item in raw.split(","):
        if not item:
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_logging(args.log)

    collectors = build_collectors(args)
    if not collectors:
        LOG.error("No collectors enabled. Check --only flag.")
        return 1

    if args.command:
        subcmd = args.command[0]
        if subcmd == "top":
            LOG.error("Interactive `top` view not yet implemented.")
            return 1
        if subcmd != "run":
            LOG.error("Unknown subcommand '%s'. Supported: run, top", subcmd)
            return 1

    output_path = None if args.no_output else args.output
    agg_cfg = AggregationConfig(
        interval=args.interval,
        output_path=output_path,
        flush_every=max(1, args.flush_every),
        fsync=args.fsync,
        extra_labels=parse_labels(args.labels),
    )
    corr_ctx = (
        CorrelationContext.disabled()
        if args.disable_correlation
        else CorrelationContext.default()
    )
    aggregator = IntervalAggregator(collectors, agg_cfg, corr_ctx)
    try:
        aggregator.run_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down collectors...")
        aggregator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
