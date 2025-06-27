#!/bin/bash

# This script runs the benchmarks for the LLM Instrumentation Framework.

python3 ../benchmarks/baseline_dump.py
python3 ../benchmarks/optimized_streaming.py
python3 ../benchmarks/compression_analysis.py
python3 ../benchmarks/memory_profiling.py
