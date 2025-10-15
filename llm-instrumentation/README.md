# LLM Instrumentation Framework

A high-performance instrumentation framework for LLM interpretability and observability.

## Objectives

- **Throughput**: Maintain ≥ 90% of un-instrumented inference speed.
- **Data rate**: Sustain ≥ 2 GB/s activation streaming to disk.
- **Compression**: Achieve ≥ 3× reduction with lossy error < 1e-3 when enabled.
- **Memory**: Keep host RAM usage ≤ 24 GB with backpressure and buffering.

## Stack

- **Runtime**: PyTorch, asyncio, threading.
- **GPU**: Optional CUDA streams and pinned buffers (see `memory/cuda_manager.py`).
- **Compression**: LZ4, Zstd, optional no-op.
- **Analysis**: Hooks for downstream causal graphs and SAE-based features.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Usage

```python
from llm_instrumentation import (
    InstrumentationFramework,
    InstrumentationConfig,
    HookGranularity,
)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = InstrumentationConfig(
    granularity=HookGranularity.ATTENTION_ONLY,
    compression_algorithm="lz4",  # or "zstd" or "none"
    target_throughput_gbps=2.0,
    max_memory_gb=24,
)

framework = InstrumentationFramework(config)
framework.instrument_model(model)

with framework.capture_activations("output.stream"):
    _ = model(torch.randint(0, 100, (1, 16)))

analysis = framework.analyze_activations("output.stream")
```

## Configuration

- `granularity` (`HookGranularity`):
  - `FULL_TENSOR`: Capture all supported layer outputs.
  - `SAMPLED_SLICES`: Randomly samples elements by `sampling_rate`.
  - `ATTENTION_ONLY`: Only layers whose names include `attn`.
  - `MLP_ONLY`: Only layers whose names include `mlp`.
- `compression_algorithm` (`str`): `"lz4"`, `"zstd"`, or `"none"`.
- `target_throughput_gbps` (`float`): Desired streaming rate for tuning.
- `max_memory_gb` (`float|None`): Budget for host buffering policies.

Refer to `docs/API.md` for full API details.

## Stream Format

Each packet:

- Header: network-endian `!HI` → `(name_len: uint16, data_len: uint32)`
- Name: UTF-8 layer/module name (`name_len` bytes)
- Data: compressed tensor bytes (`data_len` bytes)

See `docs/STREAM_FORMAT.md` for a parsing example.

## Architecture

E2E path: PyTorch forward hooks → async enqueue → compression workers → ring buffer → async file writer. See `docs/ARCHITECTURE.md`.

## Benchmarks & Performance

Run `scripts/run_benchmarks.sh` and see `docs/PERFORMANCE.md` for targets, methodology, and how to generate reports.

## Block I/O Instrumentation

### Overview

STRAP-LLM includes eBPF-based block I/O monitoring to correlate disk performance with activation streaming:

- **`scripts/tracepoints.py`**: Captures latency histograms and queue depth using stable kernel tracepoints (`block:block_rq_issue`/`block:block_rq_complete`)
- **`scripts/analyze_tracepoints.py`**: Generates summaries and PNG visualizations from persisted JSONL snapshots

### Quick Start

**Collect I/O metrics:**

```bash
sudo python3 scripts/tracepoints.py --interval 5 --output tracepoints.jsonl
```

**Analyze results:**

```bash
python3 scripts/analyze_tracepoints.py \
  --input tracepoints.jsonl \
  --output-dir ../benchmarks/systems/I-O
```

### Features

- **Low overhead**: < 1% CPU usage, ~100ns per I/O request
- **Stable ABI**: Uses kernel tracepoints (no kprobes)
- **Async persistence**: Memory-mapped JSONL writer with batch flushes
- **Log₂ histograms**: Constant memory usage at any IOPS level
- **Queue depth tracking**: In-flight request monitoring per device

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--interval` | Sampling interval (seconds) | 5.0 |
| `--output` | JSONL output file | `tracepoints.jsonl` |
| `--no-output` | Disable file output | False |
| `--flush-every` | Snapshots per flush | 12 |
| `--fsync` | Force fsync after flush | False |

### Output Format

Each JSONL line contains:

- Timestamp (Unix epoch + ISO 8601)
- Per-device latency histogram (log₂ buckets in μs)
- Per-device in-flight request count

**Example:**

```json
{
  "timestamp": 1696262400.123,
  "iso_timestamp": "2025-10-02T14:20:00.123000+00:00",
  "interval_s": 5.0,
  "latency_histogram": [
    {
      "device_name": "nvme0n1",
      "total": 45123,
      "buckets": [
        {"slot": 4, "count": 12000, "bucket_low": 16, "bucket_high": 31}
      ]
    }
  ],
  "inflight": [
    {"device_name": "nvme0n1", "count": 24}
  ]
}
```

### Documentation

See **`docs/BLOCK_IO_TRACEPOINTS.md`** for:

- Prerequisites and installation
- Detailed usage examples
- Integration with LLM workflows
- Troubleshooting guide
- Performance characteristics
- Advanced customization

## CPU & Memory Metrics

### Overview

- **`scripts/system_metrics.py`**: Engancha tracepoints `exceptions:page_fault_user` y `sched:sched_switch` para capturar fallos de página por PID, tiempo fuera de CPU y presión PSI de CPU/I/O/memoria.
- Se ejecuta como root y persiste snapshots JSONL con los campos `off_cpu_ns`, `page_faults` y `pressure` cada `N` segundos.
- La salida complementa los histogramas de latencia/colas producidos por `tracepoints.py` para correlacionar latencia de servicio con contención de CPU, swapping y presión sistémica.

### Quick Start

```bash
sudo python3 scripts/system_metrics.py --interval 5 --output system_metrics.jsonl
```

Cada línea JSON incluye `timestamp`, `iso_timestamp`, `interval_s`, un mapa `off_cpu_ns` (PID → nanosegundos fuera de CPU), `page_faults` (PID → fallos de página de usuario) y la estructura `pressure` con métricas PSI para CPU, I/O y memoria.

Para ver las muestras sólo por pantalla añade `--no-output`. Usa `--flush-every` y `--fsync` para controlar el flushing asíncrono en disco.

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--interval` | Intervalo entre snapshots (s) | 5.0 |
| `--output` | Archivo JSONL de salida | `system_metrics.jsonl` |
| `--no-output` | Deshabilita escritura a disco | False |
| `--flush-every` | Snapshots por flush | 12 |
| `--fsync` | Forzar fsync tras cada flush | False |

### Correlación

Combina `system_metrics.jsonl` y `tracepoints.jsonl` con `scripts/analyze_tracepoints.py` o cargas personalizadas en pandas para atribuir latencia a contención de CPU, fallos de página, presión PSI o I/O de disco.

## Development

- Tests: `pytest -q` in repo root or the package directory.
- Examples: `examples/basic_usage.py`.
- Contributions: PRs welcome. Keep changes focused and covered by tests.
