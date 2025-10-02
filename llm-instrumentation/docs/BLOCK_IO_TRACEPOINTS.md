# Block I/O Tracepoint Instrumentation

This guide describes how to collect block-device latency histograms and in-flight queue depth with the eBPF script located at `scripts/tracepoints.py`.

## Overview

The script attaches to the stable kernel tracepoints `block:block_rq_issue` and `block:block_rq_complete` to track request lifetimes without relying on kprobes. The eBPF program keeps:

- a hash map keyed by `(dev, sector)` that stores the submit timestamp of each request;
- a log2 histogram that aggregates completion latency by device.

User space polls both maps at a configurable interval, prints the result to the terminal, and emits structured snapshots for long-term analysis with minimal overhead.

## Prerequisites

1. Install the BCC toolkit (package names vary across distributions, e.g. `bpfcc-tools`, `python3-bpfcc` and `libbpf-tools` on Debian/Ubuntu).
2. Run the script with root privileges so it can load eBPF programs and read the block tracepoints:

   ```bash
   sudo python3 llm-instrumentation/scripts/tracepoints.py
   ```

   Ensure the Python interpreter can import the `bcc` module (either via system packages or `pip install bcc`).

## CLI Options

```
--interval <seconds>    Polling interval for both maps (default: 5.0)
--output <path>         JSONL file for persisted snapshots (default: tracepoints.jsonl)
--no-output             Disable JSONL persistence; only print to stdout
--flush-every <N>       Number of snapshots batched before flushing (default: 12)
--fsync                 Force fsync() on every flush (durable, higher overhead)
```

Snapshots are queued to a background writer thread that appends to the JSONL file, avoiding extra latency in the main sampling loop. Adjust the flush and fsync cadence according to durability requirements.

## Sample Output

Every interval the script prints current histograms and queue depth:

```
08:45:12 - Latencia I/O por dispositivo:

disk = nvme0n1
     microsegundos       : count     distribution
         0 -> 1          : 0        |                                        |
         2 -> 3          : 0        |                                        |
         4 -> 7          : 0        |                                        |
         8 -> 15         : 0        |                                        |
        16 -> 31         : 1        |*                                       |
        32 -> 63         : 27       |***************************             |
        64 -> 127        : 40       |****************************************|
       128 -> 255        : 1        |*                                       |
       256 -> 511        : 10       |**********                              |
       512 -> 1023       : 3        |***                                     |
      1024 -> 2047       : 1        |*                                       |
      2048 -> 4095       : 0        |                                        |
      4096 -> 8191       : 4        |****                                    |
      8192 -> 16383      : 0        |                                        |
     16384 -> 32767      : 2        |**                                      |
Cola I/O en vuelo (requests):
           sda -> 1
           sdb -> 1
       nvme0n1 -> 1
```

## Persisted JSONL Schema

Each snapshot written to the JSONL file uses the following structure:

```json
{
  "timestamp": 1691400312.123,
  "iso_timestamp": "2023-08-07T08:45:12.123000+00:00",
  "interval_s": 5.0,
  "latency_histogram": [
    {
      "device_id": 259000,
      "device_name": "nvme0n1",
      "total": 84,
      "buckets": [
        {"slot": 5, "count": 27, "bucket_low": 32, "bucket_high": 63},
        {"slot": 6, "count": 40, "bucket_low": 64, "bucket_high": 127}
      ]
    }
  ],
  "inflight": [
    {"device_id": 259000, "device_name": "nvme0n1", "count": 1}
  ]
}
```

- `slot` is the base-2 logarithmic bucket index.
- `bucket_low` and `bucket_high` bound the latency range in microseconds.
- `total` is the sum of counts for the device during that window.
- `inflight` reports the approximate number of outstanding requests per device.

The compact JSONL format is easy to post-process with tools like pandas, DuckDB, or jq, and can be converted to columnar formats (Parquet/Arrow) for large datasets.

## Operational Tips

- Rotate or compress the JSONL file periodically (e.g. via `logrotate`) to keep disk usage bounded.
- Re-run the script with `--no-output` for interactive troubleshooting without touching storage.
- Filter by device by modifying the keys in the eBPF hash map following the pattern from the BCC `biolatency` example, should you need to focus on specific hardware.
- The histogram map is cleared after every print to keep the cost bounded even at high request rates.
