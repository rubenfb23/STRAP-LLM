# API Documentation

This document provides a detailed overview of the API for the LLM Instrumentation Framework.

## `InstrumentationConfig`

A data class for configuring the instrumentation framework.

**Fields:**

- `granularity` (`HookGranularity`): The level of detail to capture. Options:
  - `FULL_TENSOR`: Capture all supported layer outputs.
  - `SAMPLED_SLICES`: Capture a random subset of elements (see `HookConfig.sampling_rate`).
  - `ATTENTION_ONLY`: Capture only attention layer outputs.
  - `MLP_ONLY`: Capture only MLP outputs.
- `compression_algorithm` (`str`): Compression to apply. Supported values are `"lz4"`, `"zstd"`, and `"none"`.
- `target_throughput_gbps` (`float`): The target data capture rate in GB/s.
- `max_memory_gb` (`float`): The maximum amount of host RAM to use for buffering.

**Enums:**

- `HookGranularity`:
  - `FULL_TENSOR`
  - `SAMPLED_SLICES`
  - `ATTENTION_ONLY`
  - `MLP_ONLY`

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

### `capture_activations(self, output_path: str, track_per_token: bool = False) -> ContextManager[Optional[_TokenBoundaryTracker]]`

A context manager that captures activations during inference and saves them to the specified file.

When `track_per_token=True`, an in-memory token tracker is returned and token metadata is saved to `{output_path}_tokens.json` at exit. The token tracker operates outside the hot compression/streaming path and does not impact performance.

**Parameters:**

- `output_path` (`str`): Path to the `.stream` file where activations are saved.
- `track_per_token` (`bool`, default `False`): Enable per-token boundary tracking.

**Yields:**

- `None` if `track_per_token=False`, otherwise a `_TokenBoundaryTracker` with `record_token(token_id: int, token_text: str, position: Optional[int] = None)`.

**Examples:**

```python
# Standard usage
with framework.capture_activations("output.stream"):
    _ = model.generate(input_ids, max_length=100)

# Per-token tracking usage
with framework.capture_activations("gen.stream", track_per_token=True) as tracker:
    ids = input_ids
    for _ in range(64):
        out = model(ids)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tracker.record_token(next_tok[0].item(), tokenizer.decode(next_tok[0]))
        ids = torch.cat([ids, next_tok], dim=-1)
```

### `analyze_activations(self, data_path: str) -> dict`

Loads captured activation data from a file and returns an analysis object.

**Parameters:**

- `data_path` (`str`): The path to the file containing the captured activations.

If token tracking was enabled during capture, load the adjacent `*_tokens.json` to correlate packets with tokens.

**Returns:**

- `dict`: A dictionary with keys like `packets`, `total_compressed_bytes`, `per_layer`, and others.

### `analyze_activations_with_tokens(stream_path: str, framework: InstrumentationFramework) -> dict`

Convenience helper that runs `analyze_activations` and, if present, loads `{stream_path}_tokens.json` to enrich the result with `token_metadata`, `packets_per_token`, and `bytes_per_token`.

**Parameters:**

- `stream_path` (`str`): Path to the `.stream` file.
- `framework` (`InstrumentationFramework`): Framework instance used for capture.

**Returns:**

- `dict`: Analysis dictionary with optional `token_metadata`, `packets_per_token`, `bytes_per_token`.
