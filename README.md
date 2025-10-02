# STRAP-LLM: Streaming Transformer Activation Pipeline

STRAP-LLM is a high-performance framework for capturing, streaming, and analyzing the internal activations of large language models (LLMs). It targets interpretability and observability use cases where high-throughput data pipelines must preserve ≥90% of baseline inference speed while sustaining multi-GB/s activation streaming.

The core `llm-instrumentation` library instruments PyTorch transformer models, streams activations asynchronously to disk with compression, and provides a foundation for downstream analysis.

## Highlights

- **E2E capture**: Hook → async queue → compression → ring buffer → async writer.
- **Hook granularity**: `FULL_TENSOR`, `SAMPLED_SLICES`, `ATTENTION_ONLY`, `MLP_ONLY`.
- **Compression**: `lz4`, `zstd`, and `none` with FP16 downcast before serialization.
- **Throughput-first**: Background asyncio loop and thread pool, headered packets, zero-copy buffer handoffs where possible.
- **Analysis-ready**: Stream format is documented and stable for offline parsing.

## Quickstart

1) Install

```bash
cd llm-instrumentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2) Capture activations

```python
from llm_instrumentation import InstrumentationFramework, InstrumentationConfig, HookGranularity
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"  # use a smaller model locally if needed
model = AutoModelForCausalLM.from_pretrained(model_id)
tok = AutoTokenizer.from_pretrained(model_id)

cfg = InstrumentationConfig(
    granularity=HookGranularity.ATTENTION_ONLY,
    compression_algorithm="lz4",  # or "zstd" or "none"
    target_throughput_gbps=2.0,
    max_memory_gb=24,
)
fw = InstrumentationFramework(cfg)
fw.instrument_model(model)

inputs = tok.encode("The future of AI is", return_tensors="pt")
with fw.capture_activations("output.stream"):
    _ = model.generate(inputs, max_length=64)
```

3) Inspect stream

See the documented packet format in `llm-instrumentation/docs/STREAM_FORMAT.md`.

## Targets & Scope

- **Throughput slowdown**: ≤ 10% (stretch ≤ 5%).
- **Sustained dump rate**: ≥ 2 GB/s (stretch ≥ 4 GB/s).
- **Compression ratio**: ≥ 3× with reconstruction MSE < 1e-3 for lossy modes.
- **Host RAM overhead**: ≤ 24 GB.

These are project targets; actual results depend on model, hardware, and configuration. See `llm-instrumentation/docs/PERFORMANCE.md` for benchmarks and how to run them.

## Repository Structure

- `llm-instrumentation/`
  - `src/llm_instrumentation/`: Core modules (hooks, streaming, compression, memory, analysis).
  - `docs/`: Architecture, API, performance, and stream format.
  - `examples/`: Minimal runnable examples of capture and analysis.
  - `scripts/tracepoints.py`: eBPF block I/O latency collector using kernel tracepoints.
  - `scripts/analyze_tracepoints.py`: Offline analyzer that resumes the persisted snapshots, emits summaries, and renders PNG charts.
- `benchmarks/`
  - `systems/`: Tree that stores generated artefacts. The `I-O/` subfolder is the default target for tracepoint charts.
- `TFM_InstrmntLLM-Computational.pdf`: Project proposal and requirements.
- `LICENSE`

See `llm-instrumentation/docs/BLOCK_IO_TRACEPOINTS.md` for the full workflow to collect block-device metrics and generate the derived visualisations.

## Documentation

- Package overview and usage: `llm-instrumentation/README.md`
- Architecture: `llm-instrumentation/docs/ARCHITECTURE.md`
- API: `llm-instrumentation/docs/API.md`
- Stream format: `llm-instrumentation/docs/STREAM_FORMAT.md`
- Performance workflow: `llm-instrumentation/docs/PERFORMANCE.md`
- Getting started: `llm-instrumentation/docs/GETTING_STARTED.md`

## Development

- Style: Keep changes minimal and focused. Avoid unrelated refactors.
- Tests: See `llm-instrumentation/tests`. Add focused unit tests near code changes.
- Benchmarks: Run `llm-instrumentation/scripts/run_benchmarks.sh` to generate metrics.

## Contributing

Issues and PRs are welcome. Please open a discussion for larger changes or architectural proposals.
