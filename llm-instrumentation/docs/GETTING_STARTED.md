# Getting Started with the LLM Instrumentation Framework

This guide provides instructions for setting up and using the LLM Instrumentation Framework.

## Installation

To get started, you need to set up the Python environment. The provided script will create a virtual environment and install all the necessary dependencies.

```bash
# Navigate to the scripts directory
cd llm-instrumentation/scripts

# Run the setup script
./setup_environment.sh
```

This will create a `venv` directory in `llm-instrumentation` and install the packages listed in `requirements.txt`.

## Basic Usage

The core functionality of the framework is to instrument a language model and capture its internal activations during inference. Here is a basic example of how to do this:

```python
from llm_instrumentation import InstrumentationConfig, capture_activations
from transformers import AutoModelForCausalLM

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Run inference and capture the activations to a file
with capture_activations(model, preset="balanced", output_path="output.stream") as framework:
    input_ids = ...  # Prepare your input tensors
    outputs = model.generate(input_ids, max_length=100)

# The captured data is now in "output.stream"
```

The helper defaults to the `balanced` preset, but you can pick any preset (`fast_capture`, `max`, `max_compression`, `attention_analysis`, `mlp_analysis`) or supply a custom configuration. Presets return regular `InstrumentationConfig` objects, so you can chain overrides fluently:

```python
custom = InstrumentationConfig.balanced().with_compression("zstd").with_memory_limit(16)
with capture_activations(model, config=custom, output_path="custom.stream"):
    ...
```

For a complete, runnable example, see `examples/basic_usage.py`.

## Running Benchmarks

The framework includes a suite of benchmarks to measure performance for data throughput, compression, and memory usage. To run the benchmarks:

```bash
# Navigate to the scripts directory
cd llm-instrumentation/scripts

# Run the benchmark script
./run_benchmarks.sh
```

This script will execute the different benchmark files located in the `benchmarks` directory and print the results.

## Analyzing Captured Data

Once you have captured activation data, you can use the analysis tools to interpret the model's behavior. The framework provides tools for building causal graphs and visualizing attention patterns.

```python
# Analyze the captured activation data
analysis_results = framework.analyze_activations("output.stream")

# The analysis_results object contains detailed information
# that can be used for further research and visualization.
```

For a demonstration of the analysis and interpretability features, please see `examples/interpretability_demo.py`.

## Optional: Per-token Tracking

You can enable per-token boundary tracking for generation workloads. This is opt-in and does not affect the streaming hot path.

```python
from llm_instrumentation import analyze_activations_with_tokens

with framework.capture_activations("gen.stream", track_per_token=True) as tracker:
    ids = input_ids
    for _ in range(64):
        with torch.no_grad():
            out = model(ids)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tracker.record_token(next_tok[0].item(), tokenizer.decode(next_tok[0]))
        ids = torch.cat([ids, next_tok], dim=-1)
        if next_tok[0].item() == tokenizer.eos_token_id:
            break

analysis = analyze_activations_with_tokens("gen.stream", framework)
print("packets per token:", analysis.get("packets_per_token"))

# Add checkpointing to safeguard long generations and enable resume.
with framework.capture_activations(
    "gen.stream",
    track_per_token=True,
    checkpoint_interval_tokens=128,
) as tracker:
    ...

# Resume later; tokens and stream data append seamlessly.
with framework.capture_activations(
    "gen.stream",
    track_per_token=True,
    checkpoint_interval_tokens=128,
    resume_from_checkpoint=True,
) as tracker:
    ...

# Store checkpoints somewhere else if desired.
with framework.capture_activations(
    "gen.stream",
    track_per_token=True,
    checkpoint_interval_tokens=128,
    checkpoint_path="tmp/ckpt.json",
) as tracker:
    ...
```

Notes

- If you are prototyping CPU-only or want to avoid compression cost, start with `compression_algorithm="none"`.
- Use a smaller model locally to validate the capture path before scaling.
- Checkpoints default to `{output_path}.ckpt.json` and are cleaned up automatically once the run finishes successfully.

## Further Examples

The `examples` directory contains more advanced usage scenarios, including:

*   `advanced_streaming.py`: Demonstrates more complex streaming and data handling.
*   `interpretability_demo.py`: Shows how to use the analysis tools to gain insights from the captured data.

Feel free to explore these examples to better understand the capabilities of the framework.

## Block I/O Instrumentation

For measuring storage latency and queue depth alongside model runs, the repository ships an eBPF-based helper under `scripts/tracepoints.py`. It attaches to the stable `block:block_rq_*` tracepoints, prints log2 histograms per device, and can persist JSONL snapshots for offline analysis. Run it with root privileges:

```bash
sudo python3 llm-instrumentation/scripts/tracepoints.py --interval 5 --output tracepoints.jsonl
```

Use `--no-output` to disable persistence or tune `--flush-every`/`--fsync` for durability. See `docs/BLOCK_IO_TRACEPOINTS.md` for the full CLI and data format reference.
