# System Architecture

This document provides a detailed overview of the system architecture for the LLM Instrumentation Framework.

The LLM Instrumentation framework is designed for high-performance capture and analysis of transformer model activations. The architecture is modular, allowing for flexible configuration and extension.

## Core Components

The framework consists of several core components:

- **Instrumentation Core (`llm_instrumentation.core`)**: Hooks, streaming, and compression.
    - `hooks.py`: Manages PyTorch forward hooks with configurable granularity.
    - `streaming.py`: High-throughput serializer: async queue → compression workers → ring buffer → async file writer.
    - `compression.py`: LZ4, Zstd, and no-op; FP16 downcast before serialization.
    - `profiling.py`: Utilities for tracking inference, throughput, and memory.

- **Memory Management (`llm_instrumentation.memory`)**: Efficient buffering and GPU transfer utilities.
    - `ring_buffer.py`: Thread-safe byte ring buffer for streaming packets.
    - `cuda_manager.py`: Pinned buffers and CUDA streams helper for async GPU→host transfers (integration planned).
    - `async_writer.py`: Reserved for a standalone writer abstraction; current writer is integrated in `streaming.py`.

- **Analysis Engine (`llm_instrumentation.analysis`)**: This component provides tools for offline analysis of the captured activation data.
    - `causal_graph.py`: Tools for constructing and analyzing causal graphs from the activation data to understand information flow within the model.
    - `sparse_autoencoder.py`: Functionality for training and applying sparse autoencoders to find interpretable features in the activations.
    - `visualization.py`: Tools for visualizing the results of the analysis.

- **Utilities (`llm_instrumentation.utils`)**:
    - `config.py`: Manages configuration for the instrumentation and analysis.
    - `logging.py`: Provides logging facilities.
    - `metrics.py`: For collecting and reporting performance metrics.

## Data Flow

1.  The `InstrumentationFramework` uses `hooks.py` to attach forward hooks to the specified layers of the transformer model.
2.  During the model's forward pass, the hooks capture the activation tensors.
3.  These tensors are placed into the `CudaRingBuffer`.
4.  A background worker thread retrieves tensors from the ring buffer, compresses them using one of the configured `compression` algorithms, and moves them to a host memory buffer.
5.  The async writer in `streaming.py` persists packets to disk.
6.  This entire process is asynchronous to minimize impact on inference throughput.
7.  Once the data is captured, the `analysis` modules can be used to load the data and perform various interpretability studies.
