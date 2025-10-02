# Block I/O Tracepoint Instrumentation

This guide describes how to collect block-device latency histograms and in-flight queue depth with the eBPF script located at `scripts/tracepoints.py`, and how to post-process the snapshots with `scripts/analyze_tracepoints.py`.

## Overview

The script attaches to the stable kernel tracepoints `block:block_rq_issue` and `block:block_rq_complete` to track request lifetimes without relying on kprobes. The eBPF program maintains:

- A hash map (`start`) keyed by `(dev, sector)` that stores the submit timestamp of each request
- A log₂ histogram (`latency_hist`) that aggregates completion latency by device

User space polls both maps at a configurable interval and persists the results as JSONL snapshots. The async writer uses memory-mapped I/O to minimize overhead and maximize throughput.

## Prerequisites

### System Requirements

- Linux kernel ≥ 4.7 (block tracepoints introduced)
- `debugfs` mounted at `/sys/kernel/debug`
- Root privileges for eBPF program loading

### Software Dependencies

Install the BCC toolkit:

```bash
# Ubuntu/Debian
sudo apt-get install bpfcc-tools python3-bpfcc

# Fedora/RHEL
sudo dnf install bcc-tools python3-bcc
```

### Python Environment Setup

The script automatically searches for the BCC Python module in standard locations:

- `/usr/share/bcc/python`
- `/usr/lib/python3/dist-packages`
- `/usr/lib/python3/site-packages`

If using a virtual environment, create it with system packages access:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Alternatively, set `PYTHONPATH`:

```bash
export PYTHONPATH=/usr/share/bcc/python:$PYTHONPATH
```

## Usage

### Basic Execution

Run the script with root privileges:

```bash
sudo python3 llm-instrumentation/scripts/tracepoints.py
```

The script will:

1. Validate that required tracepoints are available
2. Load the eBPF program and attach probes
3. Sample metrics every 5 seconds (default)
4. Write snapshots to `tracepoints.jsonl`
5. Run until interrupted with `Ctrl-C`

### CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--interval` | float | 5.0 | Seconds between sampling windows |
| `--output` | Path | `tracepoints.jsonl` | JSONL output file path |
| `--no-output` | flag | False | Disable file output (terminal only) |
| `--flush-every` | int | 12 | Snapshots per flush operation |
| `--fsync` | flag | False | Force fsync after each flush |

### Examples

**Custom sampling interval:**

```bash
sudo python3 llm-instrumentation/scripts/tracepoints.py --interval 10
```

**Different output location:**

```bash
sudo python3 llm-instrumentation/scripts/tracepoints.py --output /var/log/io-trace.jsonl
```

**High durability mode:**

```bash
sudo python3 llm-instrumentation/scripts/tracepoints.py --fsync --flush-every 1
```

**Monitor without persistence:**

```bash
sudo python3 llm-instrumentation/scripts/tracepoints.py --no-output
```

### Architecture

**Async Writer Thread:**

- Snapshots are queued to a background thread using `SimpleQueue`
- Writer uses memory-mapped I/O (`mmap`) to minimize buffer copies
- File grows dynamically with doubling strategy
- Batch flushes reduce syscall overhead
- Graceful shutdown truncates file to exact data size

**Performance Tuning:**

- Decrease `--interval` for higher temporal resolution (increases overhead)
- Increase `--flush-every` to reduce disk write frequency
- Enable `--fsync` only if durability is critical (adds ~5-10ms per flush)

## Output Format

### JSONL Schema

Each line in the output file represents a snapshot:

```json
{
  "timestamp": 1696262400.123,
  "iso_timestamp": "2025-10-02T14:20:00.123000+00:00",
  "interval_s": 5.0,
  "latency_histogram": [
    {
      "device_id": 271581184,
      "device_name": "nvme0n1",
      "total": 45123,
      "buckets": [
        {"slot": 4, "count": 12000, "bucket_low": 16, "bucket_high": 31},
        {"slot": 5, "count": 28000, "bucket_low": 32, "bucket_high": 63},
        {"slot": 6, "count": 5123, "bucket_low": 64, "bucket_high": 127}
      ]
    }
  ],
  "inflight": [
    {"device_id": 271581184, "device_name": "nvme0n1", "count": 24}
  ]
}
```

### Field Descriptions

**Top-level fields:**

- `timestamp`: Unix epoch timestamp (seconds, float)
- `iso_timestamp`: ISO 8601 formatted UTC timestamp
- `interval_s`: Sampling interval used for this snapshot

**Latency histogram fields:**

- `device_id`: Kernel device identifier (major << 20 | minor)
- `device_name`: Human-readable device name from `/proc/diskstats`
- `total`: Sum of all request counts for this device in the interval
- `buckets`: Array of log₂ histogram buckets

**Bucket fields:**

- `slot`: Log₂ bucket index (0, 1, 2, ...)
- `count`: Number of requests in this latency range
- `bucket_low`: Lower bound in microseconds (2^slot)
- `bucket_high`: Upper bound in microseconds (2^(slot+1) - 1)

**In-flight fields:**

- `device_id`: Kernel device identifier
- `device_name`: Human-readable device name
- `count`: Approximate number of outstanding requests

### Device Identification

Device IDs follow the Linux kernel convention:

```
device_id = (major << 20) | minor
major = device_id >> 20
minor = device_id & ((1 << 20) - 1)
```

Device names are resolved from `/proc/diskstats` at startup. The script uses memory-mapped I/O to read this file efficiently.

### Log₂ Histogram Interpretation

Each bucket represents a range of latencies in microseconds:

- Slot 0: [0, 1) μs
- Slot 1: [1, 1) μs (effectively 1 μs)
- Slot 2: [2, 3] μs
- Slot 3: [4, 7] μs
- Slot 4: [8, 15] μs
- Slot 5: [16, 31] μs
- Slot 6: [32, 63] μs
- Slot 7: [64, 127] μs
- ...and so on

This exponential bucketing provides:

- **Constant memory overhead**: O(log₂(max_latency))
- **Scalability**: Works at 100k+ IOPS
- **Precision trade-off**: Loses exact values but preserves distribution shape

The compact JSONL format is suitable for streaming ingestion and can be processed with:

- Command-line tools: `jq`, `grep`, `awk`
- DataFrame libraries: `pandas`, `polars`
- Databases: DuckDB, ClickHouse
- Columnar formats: Parquet, Arrow

## Offline Analysis

### Analyzer Script

Process persisted snapshots with the companion analyzer:

```bash
python3 llm-instrumentation/scripts/analyze_tracepoints.py \
  --input tracepoints.jsonl \
  --output-dir benchmarks/systems/I-O
```

### Generated Artifacts

The analyzer produces the following outputs for each device:

**PNG Charts:**

- `{device}_latency_hist.png`: Heatmap of latency distribution over time
- `{device}_queue_depth.png`: Time series of in-flight request count
- `{device}_requests.png`: Request rate (IOPS) per sampling interval

**Summary Statistics:**

- Total request count per device
- Approximate mean latency (from log₂ buckets)
- Distribution percentiles (p50, p90, p99)
- Peak queue depth
- Time ranges with anomalous latency

### Example Workflow

**1. Run instrumentation during workload:**

```bash
# Terminal 1: Start tracepoint collection
sudo python3 llm-instrumentation/scripts/tracepoints.py --interval 5

# Terminal 2: Run your workload (e.g., LLM inference)
python3 examples/basic_usage.py
```

**2. Analyze collected data:**

```bash
python3 llm-instrumentation/scripts/analyze_tracepoints.py \
  --input tracepoints.jsonl \
  --output-dir benchmarks/systems/I-O \
  --top 5 \
  --percentiles 50,90,99
```

**3. Review visualizations:**

```bash
ls -lh benchmarks/systems/I-O/
# nvme0n1_latency_hist.png
# nvme0n1_queue_depth.png
# nvme0n1_requests.png
```

## Integration with STRAP-LLM

The block I/O instrumentation complements LLM activation capture:

**Correlation workflow:**

1. Start tracepoint collection before inference
2. Run LLM workload with activation instrumentation
3. Stop both collectors
4. Correlate timestamps to identify I/O bottlenecks during streaming

**Use cases:**

- **Diagnose streaming stalls**: Match I/O latency spikes with activation buffer flushes
- **Optimize compression**: Identify if disk I/O limits compression throughput
- **Storage planning**: Quantify sustained write rates needed for activation dumps
- **System profiling**: Understand complete resource utilization (CPU, GPU, disk)

**Example:**

```bash
# Collect I/O metrics during inference
sudo python3 llm-instrumentation/scripts/tracepoints.py --output inference_io.jsonl &
TRACE_PID=$!

# Run instrumented inference
python3 examples/basic_usage.py

# Stop I/O collection
sudo kill -SIGINT $TRACE_PID
wait $TRACE_PID

# Analyze results
python3 llm-instrumentation/scripts/analyze_tracepoints.py \
  --input inference_io.jsonl \
  --output-dir benchmarks/systems/I-O
```

## Troubleshooting

### Tracepoints Not Available

**Error:** `Tracepoints requeridos no disponibles`

**Solutions:**

- Verify kernel version: `uname -r` (requires ≥ 4.7)
- Check tracepoint existence:

  ```bash
  ls /sys/kernel/debug/tracing/events/block/block_rq_*
  ```

- Mount debugfs if needed:

  ```bash
  sudo mount -t debugfs none /sys/kernel/debug
  ```

### BCC Import Failure

**Error:** `No se pudo importar 'bcc.BPF'`

**Solutions:**

- Install BCC packages:

  ```bash
  sudo apt-get install python3-bpfcc bpfcc-tools
  ```

- Create venv with system packages:

  ```bash
  python3 -m venv --system-site-packages .venv
  ```

- Add BCC to Python path:

  ```bash
  export PYTHONPATH=/usr/share/bcc/python:$PYTHONPATH
  ```

### Permission Errors

**Error:** `Este script necesita ejecutarse como root`

**Solutions:**

- Run with `sudo`: `sudo python3 llm-instrumentation/scripts/tracepoints.py`
- Grant CAP_BPF capability (kernel ≥ 5.8):

  ```bash
  sudo setcap cap_bpf,cap_perfmon+ep $(which python3)
  ```

### Script Hangs on Exit

**Symptoms:** `Ctrl-C` doesn't stop the script cleanly

**Solutions:**

- Force quit with `Ctrl-\` (SIGQUIT)
- Check for orphaned mmap resources:

  ```bash
  lsof | grep tracepoints.jsonl
  ```

- Kill process by PID:

  ```bash
  sudo kill -9 $(pgrep -f tracepoints.py)
  ```

### Empty or Corrupted Output

**Symptoms:** JSONL file is empty or truncated

**Solutions:**

- Enable fsync: `--fsync` (reduces risk of data loss)
- Reduce flush interval: `--flush-every 1`
- Check disk space: `df -h`
- Verify file permissions: `ls -l tracepoints.jsonl`

## Performance Characteristics

### Overhead

**eBPF kernel overhead:**

- ~100 nanoseconds per I/O request
- Negligible impact on application latency
- Atomic histogram updates prevent contention

**Userspace overhead:**

- Map iteration: ~10-50ms per interval
- JSONL encoding: ~5-20ms per snapshot
- Disk writes: amortized via batching

**Total CPU usage:** < 1% on high-throughput workloads (>100k IOPS)

### Memory Usage

**eBPF maps:**

- `start` map: ~24 bytes per in-flight request
- `latency_hist`: ~16 bytes per (device, slot) pair
- Typical: 1-10 KB total kernel memory

**Userspace:**

- Base overhead: ~256 KB (Python runtime)
- Mmap region: grows dynamically (starts at 64 KB)
- Queue buffer: ~1-10 KB (depends on flush rate)

### Disk I/O

**Write rate:**

- ~1-10 KB/s at 5s intervals
- Scales with number of active devices
- Compression potential: ~2-3× with gzip/zstd

**Optimization tips:**

- Increase `--interval` to reduce write frequency
- Use `--flush-every` to batch writes
- Store output on fast storage (SSD recommended)

## Operational Best Practices

### Production Deployment

**Log rotation:**

```bash
# /etc/logrotate.d/tracepoints
/var/log/tracepoints.jsonl {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 root root
}
```

**Systemd service:**

```ini
# /etc/systemd/system/tracepoints.service
[Unit]
Description=Block I/O Tracepoint Collector
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/tracepoints.py --output /var/log/tracepoints.jsonl
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
```

**Resource limits:**

```bash
# Prevent runaway memory usage
ulimit -m 1048576  # 1GB max memory
ulimit -f 10485760 # 10GB max file size
```

### Monitoring Recommendations

**Metrics to track:**

- Collection script uptime
- Output file growth rate
- eBPF program loaded status
- Kernel tracepoint availability

**Alert thresholds:**

- Latency p99 > 10ms (potential disk issue)
- Queue depth > 100 (saturation)
- Zero I/O activity (collection failure)

### Data Retention

**Guidelines:**

- Retain raw JSONL for 7-30 days
- Archive compressed data for 90 days
- Convert to Parquet for long-term storage
- Aggregate old data to hourly summaries

## Advanced Usage

### Custom Device Filtering

Modify the eBPF program to filter specific devices:

```c
// Add device filter before recording timestamp
TRACEPOINT_PROBE(block, block_rq_issue)
{
    // Only monitor nvme0n1 (adjust device ID)
    if (args->dev != 271581184) {
        return 0;
    }
    
    struct start_key key = {
        .dev = args->dev,
        .sector = args->sector
    };
    // ... rest of the code
}
```

### Request Size Tracking

Extend the histogram key to include request size:

```c
typedef struct disk_key {
    dev_t dev;
    u64 slot;
    u32 size_kb;  // Add size bucket
} disk_key_t;
```

### Real-Time Streaming

Replace JSONL writer with perf buffer for live dashboards:

```python
# Define perf buffer callback
def handle_event(cpu, data, size):
    event = b["events"].event(data)
    print(json.dumps(event_to_dict(event)))

# Attach callback
b["events"].open_perf_buffer(handle_event)
while True:
    b.perf_buffer_poll()
```

## Limitations and Future Work

**Current limitations:**

- No per-process/cgroup filtering
- Histogram precision limited to log₂ buckets
- Device names may not resolve in containers
- No support for device-mapper volumes

**Planned enhancements:**

- Prometheus exporter integration
- Extended BPF support (kernel ≥ 5.10)
- Request size distribution tracking
- Read/write operation separation
- Container-aware device mapping
