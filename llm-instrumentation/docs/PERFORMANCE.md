# Performance

This document provides a detailed overview of the performance benchmarks for the LLM Instrumentation Framework.

## Performance Goals

The framework is designed to meet the following performance targets:

- **Throughput Target**: Maintain ≥90% of the un-instrumented inference speed.
- **Data Capture Rate**: Sustain a data streaming rate of ≥2 GB/s.
- **Compression Ratio**: Achieve a data reduction of ≥3× with a Mean Squared Error (MSE) of <1e-3.
- **Memory Overhead**: Keep host RAM usage at or below 24 GB.
- **Latency Hiding**: Effectively use asynchronous I/O to overlap computation and data transfer, minimizing performance impact.

## Benchmarks

Performance is measured using the scripts in the `benchmarks/` directory.

- `baseline_dump.py`: Measures the performance of a simple, un-optimized data dumping approach.
- `optimized_streaming.py`: Measures the performance of the optimized, asynchronous streaming pipeline.
- `compression_analysis.py`: Analyzes the trade-offs between different compression algorithms in terms of speed and compression ratio.
- `memory_profiling.py`: Profiles the memory usage of the framework.

### Running the Benchmarks

To run the benchmarks, use the following script:

```bash
./scripts/run_benchmarks.sh
```

Alternatively, invoke individual benchmark files with `python -m` from the package root.

### Generating Reports

To generate performance reports from the benchmark results, use:

```bash
./scripts/generate_reports.sh
```

## Latest Results

Populate this table with your latest environment-specific results (GPU model, driver, CUDA, model size, granularity, compression algorithm). For reproducibility, include the exact command and commit hash used for the run.

| Metric                  | Target | Latest Result |
| ----------------------- | ------ | ------------- |
| Inference Speed         | ≥90%   | TBD           |
| Data Capture Rate (GB/s)| ≥2.0   | TBD           |
| Compression Ratio       | ≥3×    | TBD           |
| Memory Overhead (GB)    | ≤24    | TBD           |
