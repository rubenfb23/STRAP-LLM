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
from llm_instrumentation import InstrumentationFramework
from transformers import AutoModelForCausalLM

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# (Optional) Configure the instrumentation settings
# See docs/API.md for more details on configuration options
config = InstrumentationConfig(
    granularity=HookGranularity.ATTENTION_ONLY,
    compression_algorithm="lz4",
    target_throughput_gbps=2.0,
    max_memory_gb=24
)

# Initialize the framework
framework = InstrumentationFramework(config)

# Instrument the model with hooks
framework.instrument_model(model)

# Run inference and capture the activations to a file
with framework.capture_activations("output.stream"):
    # Your model generation code here
    input_ids = ... # Prepare your input tensors
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

## Further Examples

The `examples` directory contains more advanced usage scenarios, including:

*   `advanced_streaming.py`: Demonstrates more complex streaming and data handling.
*   `interpretability_demo.py`: Shows how to use the analysis tools to gain insights from the captured data.

Feel free to explore these examples to better understand the capabilities of the framework.
