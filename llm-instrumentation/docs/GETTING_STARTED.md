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
from llm_instrumentation import (
    InstrumentationFramework,
    InstrumentationConfig,
    HookGranularity,
)
from transformers import AutoModelForCausalLM

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure the instrumentation settings
config = InstrumentationConfig(
    granularity=HookGranularity.ATTENTION_ONLY,
    compression_algorithm="lz4",  # or "zstd" or "none"
    target_throughput_gbps=2.0,
    max_memory_gb=24,
)

# Initialize and instrument
framework = InstrumentationFramework(config)
framework.instrument_model(model)

# Run inference and capture the activations to a file
with framework.capture_activations("output.stream"):
    input_ids = ...  # Prepare your input tensors
    outputs = model.generate(input_ids, max_length=100)

# The captured data is now in "output.stream"
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

Notes

- If you are prototyping CPU-only or want to avoid compression cost, start with `compression_algorithm="none"`.
- Use a smaller model locally to validate the capture path before scaling.

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
