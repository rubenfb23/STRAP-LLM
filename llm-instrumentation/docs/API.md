# API Documentation

This document provides a detailed overview of the API for the LLM Instrumentation Framework.

## `InstrumentationConfig`

A data class for configuring the instrumentation framework.

**Fields:**

- `granularity` (`HookGranularity`): The level of detail to capture. Options are `ALL`, `ATTENTION_ONLY`, `FFN_ONLY`.
- `compression_algorithm` (`str`): The compression algorithm to use. Supported values are `"lz4"`, `"zstd"`, and `None`.
- `target_throughput_gbps` (`float`): The target data capture rate in GB/s.
- `max_memory_gb` (`float`): The maximum amount of host RAM to use for buffering.

**Enums:**

- `HookGranularity`:
    - `ALL`: Capture all supported layer types.
    - `ATTENTION_ONLY`: Capture only attention layer outputs.
    - `FFN_ONLY`: Capture only feed-forward network outputs.

## `InstrumentationFramework`

The main class for interacting with the instrumentation framework.

### `__init__(self, config: InstrumentationConfig)`

Initializes the framework with the given configuration.

**Parameters:**

- `config` (`InstrumentationConfig`): The configuration object.

### `instrument_model(self, model: torch.nn.Module)`

Instruments the given PyTorch model by attaching hooks to the specified layers.

**Parameters:**

- `model` (`torch.nn.Module`): The model to instrument.

### `capture_activations(self, output_path: str) -> ContextManager`

A context manager that captures activations during inference and saves them to the specified file.

**Parameters:**

- `output_path` (`str`): The path to the file where the captured activations will be saved.

**Example:**

```python
with framework.capture_activations("output.stream"):
    outputs = model.generate(input_ids, max_length=100)
```

### `analyze_activations(self, data_path: str) -> dict`

Loads captured activation data from a file and returns an analysis object.

**Parameters:**

- `data_path` (`str`): The path to the file containing the captured activations.

**Returns:**

- `dict`: A dictionary containing analysis results, such as causal graphs and sparse autoencoder features.
