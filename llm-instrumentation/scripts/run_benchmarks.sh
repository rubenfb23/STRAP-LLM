#!/bin/bash

# This script runs the benchmarks for the LLM Instrumentation Framework.

python benchmarks/baseline_dump.py
python benchmarks/optimized_streaming.py
python benchmarks/compression_analysis.py
python benchmarks/memory_profiling.py
