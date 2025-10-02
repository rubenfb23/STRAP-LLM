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

## Block I/O Instrumentation Utilities

- `scripts/tracepoints.py`: captura latencias y profundidad de cola usando tracepoints `block_rq_issue`/`block_rq_complete`, persistiéndolos como JSONL sin escribir al terminal.
- `scripts/analyze_tracepoints.py`: resume los snapshots, genera estadísticas legibles y exporta gráficas PNG dentro de `benchmarks/systems/I-O/` (configurable vía `--charts-dir`).

Consulta `docs/BLOCK_IO_TRACEPOINTS.md` para las instrucciones completas y los flags disponibles.

## Development

- Tests: `pytest -q` in repo root or the package directory.
- Examples: `examples/basic_usage.py`.
- Contributions: PRs welcome. Keep changes focused and covered by tests.
